import os
import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.utils import shuffle
from tqdm import tqdm

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

def process_data(data:str, seed:int, neg_train:bool):
    data_dir = os.path.join('data', data)
    rating = pd.read_csv(os.path.join(data_dir, 'rating_org.csv'))
    trust = pd.read_csv(os.path.join(data_dir, 'trustnetwork_org.csv'))
    print(rating.shape)
    print(trust.shape)
    
    rating = rating.dropna(how='any')
    rating = rating.drop_duplicates(['user_id','product_id'], keep='first')
    rating = rating[rating['rating'].between(1,5,'both')]
    trust = trust.dropna(how='any')
    trust = trust.drop_duplicates(keep='first')
        
    # total users
    total_users = set(trust.user_id_1.unique()).union(set(trust.user_id_2.unique())).intersection(set(rating.user_id.unique()))
    rating = rating[rating.user_id.isin(total_users)]
    trust = trust[trust.user_id_1.isin(total_users)&trust.user_id_2.isin(total_users)]
    
    total_users = set(trust.user_id_1.unique()).union(set(trust.user_id_2.unique())).intersection(set(rating.user_id.unique()))
    rating = rating[rating.user_id.isin(total_users)]
    trust = trust[trust.user_id_1.isin(total_users) & trust.user_id_2.isin(total_users)]
    
    # encode users / items
    total_users = set(trust.user_id_1.unique()).union(set(trust.user_id_2.unique())).union(set(rating.user_id.unique()))
    total_items = rating.product_id.unique()
    item_encode = dict(zip(total_items, range(len(total_items))))

    # rating user vs total users
    rating_users = set(rating.user_id.unique())
    trust_users = total_users-rating_users
    if len(trust_users): # rating user \in total user
        r_user_encode = dict(zip(rating_users, range(len(rating_users))))
        t_user_encode = dict(zip(trust_users, range(len(rating_users)+1, len(total_users)+1)))
        assert len(set(r_user_encode.keys())&set(t_user_encode.keys()))==0
        user_encode = r_user_encode | t_user_encode
    else:    
        user_encode = dict(zip(total_users, range(len(total_users))))
        
    rating['user_id'] = rating['user_id'].map(user_encode).astype(np.int32)
    trust['user_id_1'] = trust['user_id_1'].map(user_encode).astype(np.int32)
    trust['user_id_2'] = trust['user_id_2'].map(user_encode).astype(np.int32)

    rating['product_id'] = rating['product_id'].map(item_encode).astype(np.int32)
    trust['value'] = 1.0

    rating = rating[['user_id','product_id', 'rating']]
    trust = trust[['user_id_1','user_id_2']]
        
    assert max(rating['user_id'].nunique(), trust['user_id_1'].max(), trust['user_id_2'].max())==len(total_users)

    print(f"Max index of Rating user : {rating['user_id'].max()} | Max index of Total user : {max(user_encode.values())}")
    print(f"Max # of item : {max(item_encode.values())} | Min # of item : {min(item_encode.values())}")
    
    # rating matrix
    rating_matrix = sparse.csr_matrix(([1]*rating.shape[0], (rating['user_id'], rating['product_id'])))
    
    # split data
    shuffled = shuffle(rating, random_state=1)
    test_size = int(len(rating)*0.2)
    rating_test, rating_valid, rating_train = shuffled[:test_size//2], shuffled[test_size//2:test_size], shuffled[test_size:]
    assert rating.shape[0] == (rating_train.shape[0]+rating_valid.shape[0]+rating_test.shape[0]), 'Wrong Split'
    
    # negative sampling
    print("Negative Sampling...")
    if neg_train:
        rating_train = add_negs(rating_train, rating_matrix)
    rating_valid = add_negs(rating_valid, rating_matrix)
    rating_test = add_negs(rating_test, rating_matrix)
    
    # save data    
    if neg_train:
        rating_train.to_csv(os.path.join(data_dir, f'rating_train_{seed}_neg.csv'), index=False)
    else:
        rating_train.to_csv(os.path.join(data_dir, f'rating_train_{seed}.csv'), index=False)
    rating_valid.to_csv(os.path.join(data_dir, f'rating_valid_{seed}.csv'), index=False)
    rating_test.to_csv(os.path.join(data_dir, f'rating_test_{seed}.csv'), index=False)
    trust.to_csv(os.path.join(data_dir, 'trustnetwork.csv'), index=False)

    return rating_test, rating_valid, rating_train, trust

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--neg_train', action='store_true')
    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    args = get_args()
    process_data(args.dataset, args.seed, args.neg_train)