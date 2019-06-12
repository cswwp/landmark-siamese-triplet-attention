#encoding=utf-8
# import numpy as np
# import time
# axis=0
# a = np.arange(8).reshape(2,4)
# b = np.arange(8).reshape(2,4)
#
#
# time1 = time.time()
# for i in range(10000):
#     a = np.concatenate((a, b), axis=axis)
# print('concatenate time', time.time()-time1)
# print(a.shape)
# a = np.arange(8).reshape(2,4)
# b = np.arange(8).reshape(2,4)
# time2 = time.time()
# for i in range(10000):
#     a = np.vstack([a, b])
#
# print('vstack time', time.time()-time1)
# print(a.shape)
#

import faiss
from faiss import normalize_L2
import numpy as np
import time
def IndexIVFFlat():
    d = 2048                           # dimension
    nb = 1000050                    # database size
    np.random.seed(1234)             # make reproducible
    training_vectors= np.random.random((nb, d)).astype('float32')*10

    normalize_L2(training_vectors)

    nlist = 1000  # 聚类中心的个数
    k = 50 #邻居个数
    quantizer = faiss.IndexFlatIP(d)  # the other index，需要以其他index作为基础

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    # by default it performs inner-product search
    assert not index.is_trained
    t_tr = time.time()
    index.train(training_vectors)
    print('tr time:', time.time()-t_tr)
    assert index.is_trained
    index.nprobe = 300  # default nprobe is 1, try a few more
    t_s = time.time()
    index.add(training_vectors)  # add may be a bit slower as well
    print('add time:', time.time()-t_s)
    t1=time.time()
    D, I = index.search(training_vectors[:100], k)  # actual search
    t2 = time.time()
    print('faiss kmeans result times {}'.format(t2-t1))
    # print(D[:5])  # neighbors of the 5 first queries
    print(I[:5])
    topk = 5

    np.save('rank{}'.format(topk) + '.npy', I)
    np.save('similarity{}'.format(topk) + '.npy', D)
    #print(D[:5])

IndexIVFFlat()
