#encoding=utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import csr_matrix
import random

def read_csv(csv_file_path, tar_dir, root='/data5/wwp/landmark/DATAV2/train/train'):
    data = pd.read_csv(csv_file_path)
    print("Data size", data.shape)
    delete = []
    for index, row in data.iterrows():
        #print('index:', index)
        if not os.path.exists(os.path.join(root, row['id']+'.jpg')):
            delete.append(index)
        elif row['landmark_id'] == 'None':
            delete.append(index)
        else:
            if index % 1000000 == 0:
                print('index:', index)
                print('id:', row['id'])
        # print('id:', row['id'])
        # print('landmark id:', row['landmark_id'])
        #a = input()

    # print('before:', data[:10])
    # data2 = data.drop(index=[0, 1, 2])
    # print('after:', data2[:10])
    data.drop(index=delete, inplace=True)
    file_names = data['id'].tolist()
    print("write samples to txt")
    labels = data['landmark_id'].tolist()
    write_fiter_txt(file_names, labels, tar_dir)
    return file_names, labels


def write_fiter_txt(fns, labels, tar_dir):
    wob = open(os.path.join(tar_dir, 'sample.txt'), 'w')
    for i in range(len(fns)):
        temp = fns[i] + '\t' + str(labels[i]) + '\n'
        wob.write(temp)
    wob.close()



def plot_data_distribution(xlabel, data):

    # data = np.random.randn(10000)
    # print(data)
    #print(len(data))
    # data = [i for i in range(100)]
    # print(data)
    plt.bar(data, xlabel)
    # # 显示横轴标签
    # plt.xlabel("landmark id")
    # # 显示纵轴标签
    # plt.ylabel("num of id sample")
    # # 显示图标题
    # plt.title("hist")
    plt.show()
    plt.savefig('data_distribute.png')


def read_labels2indices(txt_file):
    contents = open(txt_file, 'r').readlines()
    res = {}
    for row in contents:
        lst = row.strip().split(' ')
        res[lst[0]] = np.asarray(lst[1:-1], dtype=np.int)
    return res


def write_label2indices(set_label2indices, file_txt):
    wob = open(file_txt, 'w')
    for key in set_label2indices:
        str_write = str(key)
        for index in set_label2indices[key]:
            str_write += ' '
            str_write += str(index)
        str_write += '\n'
        wob.write(str_write)
    wob.close()


def gene_train_test_labels2indices_set(traintxt, testtxt, tar_dir):
    content_tr = open(traintxt, 'r').readlines()
    content_te = open(testtxt, 'r').readlines()
    tr_f_name = []
    te_f_name = []

    tr_label = []
    te_label = []

    for row in content_tr:
        lst = row.strip().split('\t')
        #print(lst)
        tr_f_name.append(lst[0])
        tr_label.append(lst[1])

    for row in content_te:
        lst = row.strip().split('\t')
        te_f_name.append(lst[0])
        te_label.append(lst[1])

    # tr_labels_set = set(tr_label)
    # te_labels_set = set(te_label)
    ind = get_indices_sparse(np.asarray(tr_label, dtype=np.int))
    tr_label_to_indices = {i: value[0] for i, value in enumerate(ind)}

    ind = get_indices_sparse(np.asarray(te_label, dtype=np.int))
    te_label_to_indices = {i: value[0] for i, value in enumerate(ind)}

    # tr_label_to_indices = {label: np.where(np.asarray(tr_label) == label)[0]
    #                      for label in tr_labels_set}

    # te_label_to_indices = {label: np.where(np.asarray(te_label) == label)[0]
    #                      for label in te_labels_set}

    write_label2indices(tr_label_to_indices, os.path.join(tar_dir, 'tr_label2index.txt'))
    write_label2indices(te_label_to_indices, os.path.join(tar_dir, 'te_label2index.txt'))

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]



def split_tr_valid(file_names, labels, traintxt, testtxt, tar_dir, update_label2indices_txt=False, ratio=0.1, mode=1):
    # labels_set = set(labels)
    # print('labes_Set num:', len(labels_set))
    print('DATA nums:', len(labels))
    print('all sample label2indices going.......')
    if update_label2indices_txt:
        ind = get_indices_sparse(np.asarray(labels))
        label_to_indices = {i:value[0] for i, value in enumerate(ind)}
        print('size label_to_indices', len(label_to_indices))
        # label_to_indices = {label: np.where(np.asarray(labels) == label)[0]
        #                     for label in labels_set}
        write_label2indices(label_to_indices, os.path.join(tar_dir, 'landmark_id_indices.txt'))
    else:
        print("load existed label2indices>>>>>>>>")
        label_to_indices = read_labels2indices(os.path.join(tar_dir, 'landmark_id_indices.txt'))

    count = 0
    for key in label_to_indices.keys():
        count += label_to_indices[key].shape[0]
    print('count:', count)


    print("labels nums:", len(label_to_indices))
    tr_wob = open(os.path.join(tar_dir, traintxt), 'w')
    te_wob = open(os.path.join(tar_dir, testtxt), 'w')
    file_names = np.asarray(file_names, dtype=np.str)
    print('file_names:', file_names.shape)
    if mode == 0:
        for key in label_to_indices.keys():
            np.random.shuffle(label_to_indices[key])
            indices = label_to_indices[key]
            print('curr key:', key, 'curr indice:', indices)
            te_num = int(len(indices) * ratio)
            [train_indices, test_indies] = np.split(indices, [len(indices) - te_num])
            train_fn_lst = file_names[train_indices[:]]
            test_fn_lst = file_names[test_indies[:]]
            # write train and test filename and landmark id
            for fn in train_fn_lst:
                tr_wob.write(fn + '\t' + str(key) + '\n')

            for fn in test_fn_lst:
                te_wob.write(fn + '\t' + str(key) + '\n')
    else:
        key_lst = list(label_to_indices.keys())
        test_label = np.random.choice(np.asarray(key_lst), 4000, replace=False).astype(np.int)
        print('test_label:', test_label.shape)

        c1 = 0
        c2 = 0
        for key in test_label:
            test_fn_lst = file_names[label_to_indices[key]]
            assert len(test_fn_lst) == label_to_indices[key].shape[0]
            c1 += label_to_indices[key].shape[0]
            for fn in test_fn_lst:
                te_wob.write(fn + '\t' + str(key) + '\n')
        for key in label_to_indices.keys():

            if int(key) not in test_label:
                train_fn_lst = file_names[label_to_indices[key]]
                assert len(train_fn_lst) == label_to_indices[key].shape[0]
                c2 += label_to_indices[key].shape[0]
                for fn in train_fn_lst:
                    tr_wob.write(fn + '\t' + str(key) + '\n')
        print('c1:', c1, 'c2:', c2)

    tr_wob.close()
    te_wob.close()
    print('split train and test set ed')


    gene_train_test_labels2indices_set(os.path.join(tar_dir, traintxt), os.path.join(tar_dir, testtxt), tar_dir)
    print('generate train train label2indices and test label2indices ed')


def mainv1():
    csv_file_path = '/data5/wwp/landmark/csv/train.csv'
    traintxt = 'train.txt'
    testtxt = 'test.txt'
    fs, las = read_csv(csv_file_path)
    split_tr_valid(fs, las, traintxt, testtxt, update_label2indices_txt=True)

def mainv2():
    csv_file_path = '/data5/wwp/landmark/DATAV2/csv/train.csv'
    root = '/data5/wwp/landmark/DATAV2/train/train'
    tar_dir = './stage2/'
    fs, las = read_csv(csv_file_path, tar_dir, root)
    traintxt = 'train.txt'
    testtxt = 'test.txt'
    split_tr_valid(fs, las, traintxt, testtxt, tar_dir, update_label2indices_txt=True)

if __name__ == "__main__":
    mainv2()
    # # csv_file_path = '/data5/wwp/landmark/csv/train.csv'
    # # fs, las = read_csv(csv_file_path)
    # # split_tr_valid(fs, las, 'train.txt', 'test.txt')
    #
    #gene_train_test_labels2indices_set('./stage2/train.txt', './stage2/test.txt', './stage2/')
    #
    # # read_labels2indices("landmark_id_indices.txt")


