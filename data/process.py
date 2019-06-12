import numpy as np
from scipy.sparse import csr_matrix

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]



def get_indices_simple(data):
    return [np.where(data == i) for i in range(0, data.max() + 1)]



data_small = np.random.randint(0, 100, size=(100, 100, 10))
all(np.allclose(i1, i2)
    for i1, i2 in zip(get_indices_simple(data_small),
                      get_indices_sparse(data_small)))


data = np.random.randint(0, 301, size=(1000, 1000, 10))

%time ind = get_indices_simple(data)
# CPU times: user 14.1 s, sys: 638 ms, total: 14.7 s
# Wall time: 14.8 s

%time ind = get_indices_sparse(data)
# CPU times: user 881 ms, sys: 301 ms, total: 1.18 s
# Wall time: 1.18 s

%time M = compute_M(data)


