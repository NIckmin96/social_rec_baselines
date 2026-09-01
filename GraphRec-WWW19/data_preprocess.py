import os
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
import scipy.sparse as sparse
import networkx as nx

# def find_multi_hop_nodes(trust, source_node):
#     multi_nodes = []
#     G = nx.from_pandas_edgelist(trust)
    
#     # calculate spd from the source node to all other nodes
#     path_lengths = nx.single_source_dijkstra_path_length(G, source_node)
#     hop = 2
#     size = 2
#     while True:
#         nodes = [node for node, length in path_lengths.items() if length==hop]
#         nodes = np.random.choice(nodes, sizes=len(nodes)//size, replace=False).tolist()
#         if len(nodes)==0:
#             break
#         multi_nodes.extend(nodes)
#         hop+=1; size*=2
        
#     return multi_nodes

def get_multi_hop_nodes(spd_matrix, hop):
    # 1. 전체 data에서 target_val과 일치하는 마스크 생성
    mask = (spd_matrix.data == hop)
    
    # 2. 마스크를 적용해 조건에 맞는 column indices만 추출
    filtered_cols = spd_matrix.indices[mask]
    
    # 3. 각 행(row)별로 조건에 맞는 데이터가 몇 개 있는지 계산
    # np.add.reduceat을 사용하여 indptr이 가리키는 행 구간별로 mask(True=1)를 합산합니다.
    row_counts = np.add.reduceat(mask, spd_matrix.indptr[:-1])
    
    # 4. row_counts를 기준으로 filtered_cols를 분할하여 행별 리스트 생성
    split_indices = np.cumsum(row_counts)[:-1]
    result_per_row = np.split(filtered_cols, split_indices)
    
    return result_per_row

def add_negs(rating_split, rating_matrix):
    user_lst, neg_lst = [],[]
    all_items = np.arange(rating_matrix.shape[1])
    for user, group in tqdm(rating_split.groupby('user_id')):
        items = group['product_id']
        nnz = rating_matrix[user].indices
        zero_indices = np.setdiff1d(all_items, nnz)
        n = len(items)
        n_samples = max(10-n, n)
        neg_samples = list(np.random.choice(zero_indices, size=n_samples, replace=False))
        user_lst.extend([user]*n_samples)
        neg_lst.extend(neg_samples)
        
    rating_lst = [0]*len(neg_lst)
    neg_df = pd.DataFrame({'user_id':user_lst, 'product_id':neg_lst, 'rating':rating_lst})
    rating_split = pd.concat([rating_split, neg_df], axis=0)
            
    return rating_split

def process_data(data:str, seed:int, neg_train:bool, multi_hops:bool):
    print(f"Creating {data} Dataset..")
    data_dir = os.path.join('data',data)
    rating = pd.read_csv(os.path.join(data_dir,'rating_org.csv'))
    social = pd.read_csv(os.path.join(data_dir,'trustnetwork_org.csv'))

    # filter data
    print(rating.shape)
    print(social.shape)
    rating = rating.dropna(how='any')
    rating = rating.drop_duplicates(subset=['user_id','product_id'])
    rating = rating[rating.rating.between(1,5,'both')]
    social = social.dropna(how='any')
    social = social.drop_duplicates()
    print(rating.shape)
    print(social.shape)

    # filter data by user
    total_users = set(social.user_id_1.unique()).union(set(social.user_id_2.unique())).intersection(set(rating.user_id.unique()))
    rating = rating[rating.user_id.isin(total_users)]
    social = social[social.user_id_1.isin(total_users) & social.user_id_2.isin(total_users)]
    total_users = set(social.user_id_1.unique()).union(set(social.user_id_2.unique())).intersection(set(rating.user_id.unique()))
    rating = rating[rating.user_id.isin(total_users)]
    social = social[social.user_id_1.isin(total_users)*social.user_id_2.isin(total_users)]
    print(rating.shape)
    print(social.shape)
    
    total_users = set(social.user_id_1.unique()).union(set(social.user_id_2.unique())).union(set(rating.user_id.unique()))
    
    # encode user/item index
    user_map = dict(zip(total_users, range(len(total_users))))
    item_map = dict(zip(rating.product_id.unique(), range(rating.product_id.nunique())))
    # rating_map = dict(zip(sorted(rating.rating.unique().tolist()), range(0,5)))
    rating['user_id'] = rating['user_id'].map(user_map)
    rating['product_id'] = rating['product_id'].map(item_map)
    # rating['rating'] = rating['rating'].map(rating_map)
    social['user_id_1'] = social['user_id_1'].map(user_map)
    social['user_id_2'] = social['user_id_2'].map(user_map)

    # split data
    shuffled_rating = shuffle(rating, random_state=seed)
    test_size = int(len(rating)*0.2)
    rating_test, rating_valid, rating_train = shuffled_rating[:test_size//2], shuffled_rating[test_size//2:test_size], shuffled_rating[test_size:]
    # rating_test, rating_train = shuffled_rating[:test_size], shuffled_rating[test_size:]
    assert rating.shape[0] == (rating_train.shape[0]+rating_valid.shape[0]+rating_test.shape[0]), 'Wrong Split'
    
    # rating matrix
    rating_matrix = sparse.csr_matrix(([1]*rating.shape[0], (rating['user_id'], rating['product_id'])))

    # negative sampling
    print("Negative Sampling...")
    if neg_train:
        rating_train = add_negs(rating_train, rating_matrix)
    rating_valid = add_negs(rating_valid, rating_matrix)
    rating_test = add_negs(rating_test, rating_matrix)

    # create data in required shape
    data = []
    
    hist_u_train = rating_train.groupby('user_id')['product_id'].apply(list).to_dict() 
    data.append(hist_u_train)
    hist_ur_train = rating_train.groupby('user_id')['rating'].apply(list).to_dict()
    data.append(hist_ur_train)
    hist_v_train = rating_train.groupby('product_id')['user_id'].apply(list).to_dict()
    data.append(hist_v_train)
    hist_vr_train = rating_train.groupby('product_id')['rating'].apply(list).to_dict()
    data.append(hist_vr_train)

    # hist_u_valid = rating_valid.groupby('user_id')['product_id'].apply(list).to_dict()
    # data.append(hist_u_valid)
    # hist_ur_valid = rating_valid.groupby('user_id')['rating'].apply(list).to_dict()
    # data.append(hist_ur_valid)
    # hist_v_valid = rating_valid.groupby('product_id')['user_id'].apply(list).to_dict()
    # data.append(hist_v_valid)
    # hist_vr_valid = rating_valid.groupby('product_id')['rating'].apply(list).to_dict()
    # data.append(hist_vr_valid)

    # hist_u_test = rating_test.groupby('user_id')['product_id'].apply(list).to_dict()
    # data.append(hist_u_test)
    # hist_ur_test = rating_test.groupby('user_id')['rating'].apply(list).to_dict()
    # data.append(hist_ur_test)
    # hist_v_test = rating_test.groupby('product_id')['user_id'].apply(list).to_dict()
    # data.append(hist_v_test)
    # hist_vr_test = rating_test.groupby('product_id')['rating'].apply(list).to_dict()
    # data.append(hist_vr_test)

    train_user = list(rating_train.user_id.values)    
    data.append(train_user)
    train_item = list(rating_train.product_id.values)
    data.append(train_item)
    train_rating = list(rating_train.rating.values)
    data.append(train_rating)
    
    valid_user = list(rating_valid.user_id.values)
    data.append(valid_user)
    valid_item = list(rating_valid.product_id.values)
    data.append(valid_item)
    valid_rating = list(rating_valid.rating.values)
    data.append(valid_rating)
    
    test_user = list(rating_test.user_id.values)
    data.append(test_user)
    test_item = list(rating_test.product_id.values)
    data.append(test_item)
    test_rating = list(rating_test.rating.values)
    data.append(test_rating)
    # social_adj_list
    social_rev = social[['user_id_2','user_id_1']].copy().rename(columns={'user_id_1':'user_id_2','user_id_2':'user_id_1'})
    social_tmp = pd.concat([social, social_rev])
    # multi hop nodes 추가
    if multi_hops:
        # shortest path distance matrix
        spd_matrix = sparse.load_npz(os.path.join(data_dir, 'shortest_path_result.npz'))
        single_hop_node_lengths = social_tmp.groupby('user_id_1')['user_id_2'].size().to_dict()
        hop = 2
        _div = 2
        ### Codex작성 코드 ###
        while True:
            # row별 multi-hop(0번째 row는 제외)
            multi_hops_per_row = get_multi_hop_nodes(spd_matrix, hop)[1:]
            # 모든 user에 대해 multi-hop 후보가 없으면 종료
            if all(len(nodes) == 0 for nodes in multi_hops_per_row):
                break
            
            rows = []
            for uid, candidates in enumerate(multi_hops_per_row):
                if len(candidates) == 0:
                    continue
                base = single_hop_node_lengths.get(uid, 0)
                k = base // _div
                if k <= 0:
                    continue
                k = min(k, len(candidates))
                sampled = np.random.choice(candidates, size=k, replace=False)
                for v in sampled:
                    rows.append((uid, v))

            if rows:
                tmp = pd.DataFrame(rows, columns=['user_id_1', 'user_id_2'])
                social_tmp = pd.concat([social_tmp, tmp], ignore_index=True)
        # while True:
        #     multi_hops = get_multi_hop_nodes(spd_matrix, hop)[1:] # row별 multi-hop(0번째 row는 제외해야함)
        #     len_nodes = list(map(lambda x:len(x), multi_hops))
        #     if list(set(len_nodes))==[0]:
        #         break
        #     tmp = pd.DataFrame({'user_id_1':list(range(0,len(multi_hops))), 'user_id_2':multi_hops})
        #     tmp['user_id_2'] = tmp.apply(lambda x:np.random.choice(x['user_id_2'], min(len(x['user_id_2']), single_hop_node_lengths[x['user_id_1']]//_div), replace=False), axis=1)
        #     tmp = tmp.explode('user_id_2')
        #     social_tmp = pd.concat([social_tmp, tmp])
            hop+=1; _div*=2
            
    social_adj_list = social_tmp.groupby('user_id_1')['user_id_2'].unique().to_dict()
    data.append(social_adj_list)
            
    rating_list = rating.rating.unique().tolist()
    data.append(rating_list)

    with open(os.path.join(data_dir, f'dataset_{seed}.pickle'), 'wb') as f:
        pickle.dump(data, f)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg_train", type=bool, default=False)
    args = parser.parse_args()
    data = process_data(args.data, args.seed)