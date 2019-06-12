from backbone.shufflenetv2 import ShuffleNetV2
from data.datasets import BalancedBatchSampler
import torch
from torch.autograd import Variable
import torchvision
import numpy as np
import os
import math
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from network import init_network
from config import config
from data.datasets import init_data_loader, init_transform, TEST_DATA_LOADER
import csv
import time

import faiss
import numpy as np
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device_ids = [0, 1]

def read_txt():
    con_index = open('/home/wangwenpeng/work/siamese-triplet-retriveal/data/index_data.txt', 'r').readlines()
    con_query = open('/home/wangwenpeng/work/siamese-triplet-retriveal/data/query_data.txt', 'r').readlines()

    for i in range(len(con_index)):
        con_index[i] = con_index[i].strip().split('.')[0]

    for i in range(len(con_query)):
        con_query[i] = con_query[i].strip().split('.')[0]
    return np.asarray(con_index, dtype=np.str), np.asarray(con_query, dtype=np.str)



def write_submit(rank, similarity, threshold=0.6):
    '''

    :param query_fn_array: 1*m
    :param index_fn_array: 1*M
    :param rank: m*M
    :return:
    '''

    index_fn_array, query_fn_array = read_txt()
    print('query fn size:', query_fn_array.shape)
    print('index fn size:', index_fn_array.shape)

    #rank = np.load('rank100.npy')
    exclude = np.load('exclude.npy')
    print('exclude shape:', exclude.shape)
    #similarity = np.load('similarity100.npy')


    with open('submission.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['id', 'images'])
        for i in range(rank.shape[0]):
            print('max similarity:', similarity[i][0])
            print('>threashold {} len:'.format(threshold), len(rank[i][similarity[i] > threshold]))
            index_sorted_name = index_fn_array[rank[i][similarity[i] > threshold]]
            index_str = ' '.join(index_sorted_name)
            writer.writerow([query_fn_array[i], index_str])
        for i in range(exclude.shape[0]):
            writer.writerow([exclude[i], ''])

    #writer.close()
    print('write csv done')


def read_test_csv():
    con_query = open('/home/wangwenpeng/work/siamese-triplet-retriveal/data/query_data.txt', 'r').readlines()
    for i in range(len(con_query)):
        con_query[i] = con_query[i].strip().split('.')[0]

    con = open('/data5/wwp/landmark/csv/test.csv').readlines()
    count = 0
    lst = []
    for i in range(1, len(con)):
        row = con[i]
        value = row.split(',')[0].replace('"', '')

        if value not in con_query:
            print('value:', value   )
            count += 1
            print('count:', count)
            lst.append(value)
    np.save('exclude.npy', np.asarray(lst, dtype=np.str))

    return lst




def calculate_rank(index_array, query_array, tar_dir, topk=100):
    # index_array = np.load('index_feature.npy')
    index_fn_array = np.load('index_fn_array.npy')
    # query_array = np.load('query_feature.npy')
    query_fn_array = np.load('query_fn_array.npy')

    print('load done')

    print('index array shape:', index_array.shape)
    print('query array shape:', query_array.shape)

    d = 2048  # dimension
    nlist = 2000  # 聚类中心的个数
    quantizer = faiss.IndexFlatIP(d)  # the other index，需要以其他index作为基础

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    # by default it performs inner-product search
    assert not index.is_trained
    t_tr = time.time()
    index.train(index_array)
    print('tr time:', time.time() - t_tr)
    assert index.is_trained
    index.nprobe = 300  # default nprobe is 1, try a few more
    t_s = time.time()
    index.add(index_array)  # add may be a bit slower as well
    print('add time:', time.time() - t_s)
    t1 = time.time()
    D, I = index.search(query_array, topk)  # actual search
    t2 = time.time()
    print('faiss kmeans result times {}'.format(t2 - t1))
    # print(D[:5])  # neighbors of the 5 first queries
    print(I[:5])
    print(D[:5])

    np.save(os.path.join(tar_dir, 'rank{}'.format(topk)+'.npy'), I)
    np.save(os.path.join(tar_dir, 'similarity{}'.format(topk)+'.npy'), D)
    print('save rank done')
    write_submit(query_fn_array, index_fn_array, I)

    print('done')




def inference_test(args):
    tar_dir = time.strftime("%m %d %H:%M:%S %Y", time.localtime())
    os.mkdir(tar_dir)

    mode_path = os.path.join(args.directory, args.resume)
    params = {'architecture': args.arch, 'pooling': args.pool}
    model = init_network(params)
    if not os.path.exists(mode_path):
        print(">> No checkpoint found at '{}'".format(mode_path))
        return
    else:
        # load checkpoint weights and update model and optimizer
        print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
        checkpoint = torch.load(mode_path)
        start_epoch = checkpoint['epoch']
        print('ul epoch:', start_epoch)
        min_loss = checkpoint['min_loss']
        print(min_loss)
        model.load_state_dict(checkpoint['state_dict'])
        print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    model = torch.nn.DataParallel(model, device_ids=device_ids, dim=0).cuda()  # Encapsulate the model
    model.eval()
    cuda = args.cuda
    batch_size = 16
    index_data_loader, query_data_loader = TEST_DATA_LOADER(kwargs = {'num_workers': 16, 'pin_memory': False, 'batch_size':batch_size} if cuda else {})
    print('steps num:', len(index_data_loader))
    count = 0
    index_array_embedding = np.empty([0, 2048])
    index_array_fn = np.empty(0)

    for index_tensor, index_fn in index_data_loader:
        print('count:', count)
        count += 1
        print(index_tensor.shape, len(index_fn))
        index_tensor = index_tensor.cuda()
        t_start = time.time()
        out_tesnor = model.forward(index_tensor)
        print("foward time:", time.time() - t_start)

        t_start = time.time()
        out_tesnor = out_tesnor.cpu().detach().numpy()
        print('gpu2cpu time:', time.time() - t_start)
        index_array_embedding = np.concatenate((index_array_embedding, out_tesnor), axis=0)
        print(index_fn[0:10])
        index_array_fn = np.concatenate((index_array_fn, index_fn))
        print('index_array_fn shape:', index_array_fn.shape)
    print('index_array_embedding shape:', index_array_embedding.shape)
    print('index_array_fn shape:', index_array_fn.shape)
    np.save(os.path.join(tar_dir, 'index_array_embedding.npy'), index_array_embedding)
    np.save(os.path.join(tar_dir, 'index_array_fn.npy'), index_array_fn)
    print('save index embedding and fn done')

    count = 0
    all_sum = len(query_data_loader)
    print('allsum', all_sum)
    query_array_embeeding = np.empty(0, 2048)
    query_array_fn = np.empty(0)
    for query_tensor, query_fn in query_data_loader:
        print('allsum', all_sum, '   count:', count)
        count += 1
        query_tensor = query_tensor.cuda()
        out_tesnor = model.forward(query_tensor)
        out_tesnor = out_tesnor.cpu().detach().numpy()
        query_array_embeeding = np.concatenate((query_array_embeeding, out_tesnor))
        query_array_fn = np.concatenate((query_array_fn, query_fn))

    print('query_array_embeeding shape:', query_array_embeeding.shape)
    print('query_array_fn shape:', query_array_fn.shape)
    np.save(os.path.join(tar_dir, 'query_array_embeeding.npy'), query_array_embeeding)
    np.save(os.path.join(tar_dir, 'query_fn_array.npy'), query_fn_array)
    print('save query embedding and fn done')
    calculate_rank(index_array_embedding, query_array_embeeding, tar_dir, topk=100)


if __name__ == '__main__':
    #calculate_rank()
    write_submit('')
    #read_test_csv()