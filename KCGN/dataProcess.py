import pickle 
import numpy as np
import scipy.sparse as sp
import random
import os
import argparse
from create_adj import creatMultiItemUserAdj
import networkx as nx


def splitData(dataset, seed, cv, regen_base=False):
    print("splitData")
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    output_paths = [
        os.path.join(DIR, "train.pkl"),
        os.path.join(DIR, "train_time.pkl"),
        os.path.join(DIR, "test.pkl"),
        os.path.join(DIR, "valid.pkl"),
        os.path.join(DIR, "trust.pkl"),
        os.path.join(DIR, "category.pkl"),
    ]
    if (not regen_base) and all(os.path.isfile(p) for p in output_paths):
        with open(output_paths[0], 'rb') as fs:
            train = pickle.load(fs)
        with open(output_paths[1], 'rb') as fs:
            train_time = pickle.load(fs)
        with open(output_paths[2], 'rb') as fs:
            test = pickle.load(fs)
        with open(output_paths[3], 'rb') as fs:
            valid = pickle.load(fs)
        with open(output_paths[4], 'rb') as fs:
            trust = pickle.load(fs)
        with open(output_paths[5], 'rb') as fs:
            category = pickle.load(fs)
        return train, train_time, test, valid, trust, category
    
    with open(DIR + "/category.pkl", 'rb') as fs:
        category = pickle.load(fs)
    with open(DIR + "/ratings.pkl", 'rb') as fs:
        data = pickle.load(fs)
    with open(DIR + "/times.pkl", 'rb') as fs:
        time = pickle.load(fs)
    with open(DIR + "/trust.pkl", 'rb') as fs:
        trust = pickle.load(fs)
    assert np.sum(data.tocoo().row != time.tocoo().row) == 0
    assert np.sum(data.tocoo().col != time.tocoo().col) == 0
    row, col = data.shape
    print("user num = %d, item num = %d"%(row, col))

    train_row, train_col, train_data, train_data_time = [], [], [], []
    test_row, test_col, test_data = [], [], []
    valid_row, valid_col, valid_data = [], [], []

    # userList = np.where(np.sum(data!=0, axis=1)>=2)[0]
    for i in range(row):
        tmp_data = data[i].toarray()[0]
        if np.sum(tmp_data != 0) < 10:
            continue
        tmp_data_time = time[i].toarray()[0]
        uid = [i] * col 
        num = data[i].nnz # 0이 아닌 element의 개수
        #降序排序
        idx = np.argsort(-tmp_data_time).tolist() # timestamp descending order
        idx = idx[: num] # 0이 아닌 데이터만 Filter
        rating_data = tmp_data[idx].tolist() # 정렬된 timestamp 순서로 rating data 추출
        time_data = tmp_data_time[idx].tolist() # time data 순서대로 추출
        # all non-zero인지 확인
        assert np.sum(tmp_data[idx] == 0) == 0
        assert np.sum(tmp_data_time[idx] == 0) == 0
        
        test_num = int(num*0.1)
        train_num = num - 2*test_num

        test_row += [uid[0]]*test_num
        test_col += idx[:test_num]
        test_data += rating_data[:test_num]

        valid_row += [uid[1]]*test_num
        valid_col += idx[test_num:2*test_num]
        valid_data += rating_data[test_num:2*test_num]

        train_row += uid[0: train_num]
        train_col += idx[2*test_num:]
        train_data += rating_data[2*test_num:]
        train_data_time += time_data[2*test_num:]
        assert (0 in train_data) == False
        assert (0 in train_data_time) == False


    train = sp.csc_matrix((train_data, (train_row, train_col)), shape=data.shape)
    train_time = sp.csc_matrix((train_data_time, (train_row, train_col)), shape=data.shape)

    test  = sp.csc_matrix((test_data, (test_row, test_col)), shape=data.shape)
    valid  = sp.csc_matrix((valid_data, (valid_row, valid_col)), shape=data.shape)

    print("train num = %d, train rate = %.2f"%(train.nnz, train.nnz/data.nnz))
    print("test num = %d, test rate = %.2f"%(test.nnz, test.nnz/data.nnz))
    print("valid num = %d, valid rate = %.2f"%(valid.nnz, valid.nnz/data.nnz))

    with open(DIR + "/train.pkl", 'wb') as fs:
        pickle.dump(train.tocsr(), fs)
    with open(DIR + "/train_time.pkl", 'wb') as fs:
        pickle.dump(train_time.tocsr(), fs)

    with open(DIR + "/test.pkl", 'wb') as fs:
        pickle.dump(test.tocsr(), fs)
    with open(DIR + "/valid.pkl", 'wb') as fs:
        pickle.dump(valid.tocsr(), fs)

    with open(DIR + "/trust.pkl", 'wb') as fs:
        pickle.dump(trust.tocsr(), fs)
    with open(DIR + "/category.pkl", 'wb') as fs:
        pickle.dump(category.tocsr(), fs)

    return train, train_time, test, valid, trust, category


def filterData(dataset, seed, cv, regen_base=False):
    print("filterData")
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    output_paths = [
        os.path.join(DIR, "train.pkl"),
        os.path.join(DIR, "test.pkl"),
        os.path.join(DIR, "valid.pkl"),
        os.path.join(DIR, "train_time.pkl"),
        os.path.join(DIR, "trust.pkl"),
        os.path.join(DIR, "category.pkl"),
    ]
    if (not regen_base) and all(os.path.isfile(p) for p in output_paths):
        with open(output_paths[0], 'rb') as fs:
            train = pickle.load(fs)
        with open(output_paths[1], 'rb') as fs:
            test = pickle.load(fs)
        with open(output_paths[2], 'rb') as fs:
            valid = pickle.load(fs)
        with open(output_paths[3], 'rb') as fs:
            train_time = pickle.load(fs)
        with open(output_paths[4], 'rb') as fs:
            trust = pickle.load(fs)
        with open(output_paths[5], 'rb') as fs:
            category = pickle.load(fs)
        return train, test, valid, train_time, trust, category
    
    #filter
    with open(DIR + "/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/test.pkl", 'rb') as fs:
        test = pickle.load(fs)
    with open(DIR + "/valid.pkl", 'rb') as fs:
        valid = pickle.load(fs)
    with open(DIR + "/category.pkl", 'rb') as fs:
        category = pickle.load(fs)

    with open(DIR + "/train_time.pkl", 'rb') as fs:
        train_time = pickle.load(fs)

    with open(DIR + "/trust.pkl", 'rb') as fs:
        trust = pickle.load(fs)
    trust = trust + trust.transpose()
    trust = (trust != 0)*1
    a = np.sum(np.sum(train != 0, axis=1) ==0) # user에 대해서 interaction이 전혀 없는 경우
    b = np.sum(np.sum(train != 0, axis=0) ==0) # item에 대해서 interaction이 전혀 없는 경우
    c = np.sum(np.sum(trust, axis=1) == 0) # link가 없는 user의 존재 여부
    while a != 0 or b != 0 or c != 0:
        if a != 0: # interaction이 없는 user가 존재하는 경우 filter
            idx, _ = np.where(np.sum(train != 0, axis=1) != 0)
            train = train[idx]
            test = test[idx]
            valid = valid[idx]
            train_time = train_time[idx]
            trust = trust[idx][:, idx]
        elif b != 0: # interaction이 없는 item이 존재하는 경우 filter
            _, idx = np.where(np.sum(train != 0, axis=0) != 0)
            train = train[:, idx]
            test = test[:, idx]
            valid = valid[:, idx]
            train_time = train_time[:, idx]
            category = category[idx]
        elif c != 0: # social link가 없는 user가 존재하는 경우 filter
            idx, _ = np.where(np.sum(trust, axis=1) != 0)
            train = train[idx]
            test = test[idx]
            valid = valid[idx]
            train_time = train_time[idx]
            trust = trust[idx][:, idx]
        a = np.sum(np.sum(train != 0, axis=1) ==0)
        b = np.sum(np.sum(train != 0, axis=0) ==0)
        c = np.sum(np.sum(trust, axis=1) == 0)

    nums = train.nnz+test.nnz+valid.nnz
    print("train num = %d, train rate = %.2f"%(train.nnz, train.nnz/nums))
    print("test num = %d, test rate = %.2f"%(test.nnz, test.nnz/nums))
    print("valid num = %d, valid rate = %.2f"%(valid.nnz, valid.nnz/nums))

    with open(DIR + "/train.pkl", 'wb') as fs:
        pickle.dump(train, fs)
    with open(DIR + "/test.pkl", 'wb') as fs:
        pickle.dump(test, fs)
    with open(DIR + "/valid.pkl", 'wb') as fs:
        pickle.dump(valid, fs)
    with open(DIR + "/train_time.pkl", 'wb') as fs:
        pickle.dump(train_time, fs)
    with open(DIR + "/trust.pkl", 'wb') as fs:
        pickle.dump(trust, fs)
    with open(DIR + "/category.pkl", 'wb') as fs:
        pickle.dump(category, fs)

    return train, test, valid, train_time, trust, category

def splitAgain(dataset, seed, cv, regen_base=False):
    print("splitAgain")
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    output_paths = [
        os.path.join(DIR, "train.pkl"),
        os.path.join(DIR, "test.pkl"),
        os.path.join(DIR, "train_time.pkl"),
    ]
    if (not regen_base) and all(os.path.isfile(p) for p in output_paths):
        with open(output_paths[0], 'rb') as fs:
            train = pickle.load(fs)
        with open(output_paths[1], 'rb') as fs:
            test = pickle.load(fs)
        with open(output_paths[2], 'rb') as fs:
            train_time = pickle.load(fs)
        return train, test, train_time
    
    with open(DIR + "/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/test.pkl", 'rb') as fs:
        test = pickle.load(fs)

    with open(DIR + "/train_time.pkl", 'rb') as fs:
        train_time = pickle.load(fs)

    train = train.tolil()
    test = test.tolil()
    train_time = train_time.tolil()
    
    idx = np.where(np.sum(test!=0, axis=1).A == 0)[0] # testset에서 interaction이 없는 user index
    # test data에서 interaction이 없는 user에 대해서 train data에서 interaction이 2개 이상 존재하는 경우, 
    for i in idx:
        uid = i
        tmp_data = train[i].toarray()[0]
        if np.sum(tmp_data != 0) < 2: 
            continue
        num = train[i].nnz
        tmp_data_time = train_time[i].toarray()[0]
        l = np.argsort(-tmp_data_time).tolist()
        l = l[: num]
        # test[uid, l[0]] = train[uid, l[0]]
        # test data의 가장 마지막 sequence에 대한 평가는 1로 하고, train data의 가장 마지막 sequence item에 대한 평가는 0으로 바꿈
        data = train[uid, l[0]]
        test[uid, l[0]] = data
        train[uid, l[0]] = 0
        train_time[uid, l[0]] = 0
        # => test에서 interaction이 없는 user가 존재하는 경우, train의 마지막 interaction을 Test로 옮김
    
    train = train.tocsr()
    train_time = train_time.tocsr()
    test = test.tocsr()
    assert  np.sum(train.tocoo().data == 0)==0
    assert  np.sum(test.tocoo().data == 0)==0
    assert  (train+test).nnz == train.nnz+test.nnz

    with open(DIR + "/train.pkl", 'wb') as fs:
        pickle.dump(train, fs)
    with open(DIR + "/test.pkl", 'wb') as fs:
        pickle.dump(test, fs)
    with open(DIR + "/train_time.pkl", 'wb') as fs:
        pickle.dump(train_time, fs)

    return train, test, train_time


def generateGraph(dataset, seed, regen_seed=False):
    print("generateGraph")
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    output_path = os.path.join(DIR, f"uu_vv_graph_{seed}.pkl")
    if (not regen_seed) and os.path.isfile(output_path):
        with open(output_path, 'rb') as fs:
            return pickle.load(fs)
        
    with open(DIR + "/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/trust.pkl", 'rb') as fs:
        trustMat = pickle.load(fs)
    with open(DIR + "/category.pkl", 'rb') as fs:
        categoryMat= pickle.load(fs)
    with open(DIR + "/categoryDict.pkl", 'rb') as fs:
        categoryDict = pickle.load(fs)
    
    userNum, itemNum =  train.shape
    assert categoryMat.shape[0] == train.shape[1]
    mat = (trustMat.T + trustMat) + sp.eye(userNum) 
    UU_mat = (mat != 0)*1 # trust marix 생성(1/0)

    ITI_mat = sp.dok_matrix((itemNum, itemNum))
    categoryMat = categoryMat.toarray()
    for i in range(categoryMat.shape[0]): # item 별로,
        itemTypeList = np.where(categoryMat[i])[0] # 해당되는 feature index 추출
        for itemType in itemTypeList:
            itemList = categoryDict[itemType] # feature에 해당하는 item list 추출(자기 자신 포함)
            itemList = np.array(itemList)
            # item list의 개수에 따라 rate 차등 부여
            if itemList.size < 100:
                rate = 0.1
            elif itemList.size < 1000:
                rate = 0.01
            else:
                rate = 0.001
            # category에 해당하는 item list에서 item list의 크기에 비례해 maximum 4개의 item을 random sampling => "why not all ????"
            # 메모리 상의 이유인지 잘 모르겠지만, 아이템 하나당 최대 4개(~9999개)의 동일한 feature를 갖는 item을 mapping
            itemList2 = np.random.choice(itemList, size=int(itemList.size*rate/2), replace=False)
            itemList2 = itemList2.tolist()
            tmp = [i for _ in range(len(itemList2))] # sample의 개수만큼 item index list생성
            ITI_mat[tmp, itemList2] = 1 # 해당 item과 feature index의 원소=1

    ####### item별로 공통된 feature를 가지고 있는지에 대한 item-matrix #######
    ITI_mat = ITI_mat.tocsr()
    ITI_mat = ITI_mat + ITI_mat.T + sp.eye(itemNum)
    ITI_mat = (ITI_mat != 0)*1

    uu_vv_graph = {}
    uu_vv_graph['UU'] = UU_mat
    uu_vv_graph['II'] = ITI_mat
    with open(DIR + f'/uu_vv_graph_{seed}.pkl', 'wb') as fs:
        pickle.dump(uu_vv_graph, fs)

    return uu_vv_graph

    
def createCategoryDict(dataset, seed, cv, regen_base=False):
    print("createCategoryDict")
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    output_path = os.path.join(DIR, "categoryDict.pkl")
    if (not regen_base) and os.path.isfile(output_path):
        with open(output_path, 'rb') as fs:
            return pickle.load(fs)
        
    with open(DIR + "/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/category.pkl", 'rb') as fs:
        category = pickle.load(fs)
    
    assert category.shape[0] == train.shape[1]
    categoryDict = {}
    categoryData = category.toarray()
    for i in range(categoryData.shape[0]):
        iid = i
        typeList = np.where(categoryData[i])[0]
        # typeid = categoryData[i]
        for typeid in typeList:
            if typeid in categoryDict:
                categoryDict[typeid].append(iid)
            else:
                categoryDict[typeid] = [iid]
    with open(DIR + "/categoryDict.pkl", 'wb') as fs:
        pickle.dump(categoryDict, fs)

    return categoryDict

def testNegSample(dataset, seed, cv, regen_seed=False):
    print("testNegSample")
    DIR = os.path.join(os.getcwd(), "dataset", dataset)
    test_path = os.path.join(DIR, f"test_data_{seed}.pkl")
    valid_path = os.path.join(DIR, f"valid_data_{seed}.pkl")
    if (not regen_seed) and os.path.isfile(test_path) and os.path.isfile(valid_path):
        with open(test_path, 'rb') as fs:
            test_data = pickle.load(fs)
        with open(valid_path, 'rb') as fs:
            valid_data = pickle.load(fs)
        return test_data, valid_data
    
    #filter
    with open(DIR + "/train.pkl", 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/test.pkl", 'rb') as fs:
        test = pickle.load(fs)
    with open(DIR + "/valid.pkl", 'rb') as fs:
        valid = pickle.load(fs)

    train = train.todok()
    test_u = test.tocoo().row
    test_v = test.tocoo().col
    test_r = test.tocoo().data
    
    valid_u = valid.tocoo().row
    valid_v = valid.tocoo().col
    valid_r = valid.tocoo().data
    assert test_u.size == test_v.size
    assert valid_u.size == valid_v.size
    n = test_u.size
    test_data = []
    for i in range(n):
        u = test_u[i]
        v = test_v[i]
        r = test_r[i]
        test_data.append([u, v, r])
        for t in range(100):
            j = np.random.randint(test.shape[1])
            while (u, j) in train or j == v:
                j = np.random.randint(test.shape[1])
            test_data.append([u, j, 0])
    
    n = valid_u.size
    valid_data = []
    for i in range(n):
        u = valid_u[i]
        v = valid_v[i]
        r = valid_r[i]
        valid_data.append([u, v, r])
        for t in range(100):
            j = np.random.randint(valid.shape[1])
            while (u, j) in train or j == v:
                j = np.random.randint(valid.shape[1])
            valid_data.append([u, j, 0])
    
    with open(DIR + f"/test_data_{seed}.pkl", 'wb') as fs:
        pickle.dump(test_data, fs)
    with open(DIR + f"/valid_data_{seed}.pkl", 'wb') as fs:
        pickle.dump(valid_data, fs)

    return test_data, valid_data


def main(dataset, seed, cv=1, regen_base=False, regen_seed=False):
    train, train_time, test, valid, trust, category = splitData(dataset, seed, cv, regen_base=regen_base)
    train, test, valid, train_time, trust, category = filterData(dataset, seed, cv, regen_base=regen_base)
    train, test, train_time = splitAgain(dataset, seed, cv, regen_base=regen_base)
    train, test, valid, train_time, trust, category = filterData(dataset, seed, cv, regen_base=regen_base)
    test_data, valid_data = testNegSample(dataset, seed, cv, regen_seed=regen_seed)
    categoryDict = createCategoryDict(dataset, seed, cv, regen_base=regen_base)
    multi_adj_path = os.path.join(os.getcwd(), "dataset", dataset, "multi_item_adj.pkl")
    if (not regen_base) and os.path.isfile(multi_adj_path):
        with open(multi_adj_path, 'rb') as fs:
            multi_adj_time = pickle.load(fs)
    else:
        multi_adj_time = creatMultiItemUserAdj(dataset, seed)
    uu_vv_graph = generateGraph(dataset, seed, regen_seed=regen_seed)

    return train, test_data, valid_data, train_time, trust, uu_vv_graph, multi_adj_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #dataset params
    parser.add_argument('--dataset', type=str, default="Epinions", help="CiaoDVD,Epinions,Douban")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv', type=int, default=1, help="1,2,3,4,5")
    args = parser.parse_args()

    dataset = args.dataset+ "_time"
    dataset = args.dataset

    splitData(dataset, args.seed, args.cv)
    filterData(dataset, args.seed, args.cv)
    splitAgain(dataset, args.seed, args.cv)
    filterData(dataset, args.seed, args.cv)

    testNegSample(dataset, args.seed, args.cv)

    createCategoryDict(dataset, args.seed, args.cv)
    creatMultiItemUserAdj(dataset, args.seed)
    generateGraph(dataset, args.seed)
    
    print("Done")
