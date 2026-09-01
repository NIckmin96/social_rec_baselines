import numpy as np
import pickle 
import scipy.sparse as sp
import os

from ToolScripts.utils import loadData

def creatMultiItemUserAdj(dataset, seed=None):
    trainMat, _, _, trainMat_time, _ = loadData(dataset, seed=seed)

    ratingClass = np.unique(trainMat.data).size
    userNum, itemNum = trainMat.shape
    multi_adj = sp.lil_matrix((ratingClass*itemNum, userNum), dtype=np.int64) # 같은 item이어도 rating class(1~5) 별로 서로 다른 row를 가짐
    uidList = trainMat.tocoo().row
    iidList = trainMat.tocoo().col
    rList = trainMat.tocoo().data
    # time = trainMat_time.tocoo().data

    for i in range(uidList.size):
        uid = uidList[i]
        iid = iidList[i]
        r = rList[i]
        multi_adj[iid*ratingClass+r-1, uid] = trainMat_time[uid, iid] # multi_adj matrix에 (u,i) pair에 대한 interaction이 언제 일어났는지에 대한 정보를 rating값에 맞춰 저장
        assert trainMat_time[uid, iid] != 0

    a = sp.csr_matrix((multi_adj.shape[1], multi_adj.shape[1])) # u x u
    b = sp.csr_matrix((multi_adj.shape[0], multi_adj.shape[0])) # i*r(5) x i*r(5)
    multi_adj2 = sp.vstack([sp.hstack([a, multi_adj.T]), sp.hstack([multi_adj,b])]) # hstack (1) : (u, u+i*5), hstack(2) : (i*5, i*5 + u) -> vstack : (u+i*5, u+i*5)

    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    path = DIR + '/multi_item_adj.pkl'
    multi_adj2 = multi_adj2.tocsr()
    with open(path, 'wb') as fs:
        pickle.dump(multi_adj2, fs)
    print("create multi_item_feat")

    return multi_adj2



# if __name__ == '__main__':
#     creatMultiItemUserAdj("CiaoDVD_time", 1)
#     print("Done")
