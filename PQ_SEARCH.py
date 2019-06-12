import nanopq
import numpy as np



def extract_embeddings(net, input_tesnor):
    net.eval()
    out_put = net.foward(input_tesnor)
    out_put = out_put.cpu()
    return out_put



def pq_search(source_dataset, query_vector):
    pq = nanopq.PQ(M=8)
    pq.fit(source_dataset)

    source_code = pq.encode(source_dataset)

    dists = pq.dtable(query_vector).adist(source_code)  # (10000, )

    print(dists)



