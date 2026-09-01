import os
import time
import random
import argparse
import numpy as np
import pandas as pd
# import torch.backends
from tqdm import tqdm
import scipy.sparse as sparse
# from scipy.sparse import dok_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_preprocess.metrics import rmse as _rmse, mae as _mae, dump_predictions  # noqa: E402

from utils import *
from data_preprocess import *
from TrustSVD import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='single')
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=float, default=10)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--regen", action='store_true')
    parser.add_argument("--neg_train", action='store_true')
    args = parser.parse_args()
    
    return args

def load_data(data_path, seed, neg_train=False, regen=False):
    train_dir = os.path.join(data_path, f'rating_train_{seed}.csv')
    valid_dir = os.path.join(data_path, f'rating_valid_{seed}.csv')
    test_dir = os.path.join(data_path, f'rating_test_{seed}.csv')
    trust_dir = os.path.join(data_path, 'trust_matrix.npz')
    
    if (os.path.isfile(train_dir)&os.path.isfile(valid_dir)&os.path.isfile(test_dir)&os.path.isfile(trust_dir)) and not regen:
        rating_train = pd.read_csv(train_dir)
        rating_valid = pd.read_csv(valid_dir)
        rating_test = pd.read_csv(test_dir)
        trust_matrix = sparse.load_npz(trust_dir)
    else:
        rating_test, rating_valid, rating_train, trust_matrix = process_data(data_path, seed, neg_train)
        
    return rating_train, rating_valid, rating_test, trust_matrix

def prepare_data(data, bs):
    # dataset & dataloader
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=2)
    return dataset, dataloader

def run(num_epochs, model, optimizer, device, train_loader, valid_loader, test_loader, trust_matrix, k=10):
    model = model.to(device)
    model.train()
    train_loss = []
    best_rmse = 1e10
    stop_cnt = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        tqdm_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step,batch in enumerate(tqdm_iterator):
            batch = {k:v.to(device) for k,v in batch.items()}
            r_target = batch['rating']
            rows = batch['user_id'].detach().unsqueeze(1).cpu()
            cols = batch['t_u'].detach().cpu()
            t_target = torch.from_numpy(trust_matrix[rows, cols].toarray()).float().to(device)
            r_pred, t_pred, reg_term = model(batch)
            r_pred = r_pred.squeeze()
            
            # rating loss
            r_loss = F.mse_loss(r_pred, r_target, reduction='sum')
            # trust loss
            t_mask = (batch['t_u'] != 0)
            t_loss = F.mse_loss(t_pred, t_target, reduction='none')
            t_loss = (t_loss * t_mask).sum()

            loss = 0.5*r_loss + 0.5*lambda_t*t_loss + reg_term
            
            # backpropagation & param update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            tqdm_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)" % (step, len(tqdm_iterator), loss.item()))

        running_loss/=(step+1)
        train_loss.append(running_loss)
        # validation
        best_rmse, rmse, mae, stop_cnt, neg_ndcg, neg_precision = test(model, device, valid_loader, best_rmse, stop_cnt, k=k)
        
        print(f"{epoch+1}/{num_epochs} Epochs || Best RMSE : {best_rmse:.4f} || Train Loss : {running_loss:.4f} || Valid RMSE : {rmse:.4f}")
        if stop_cnt==5:
            break

    # Eval(test)
    best_rmse, rmse, mae, stop_cnt, neg_ndcg, neg_precision = test(model, device, test_loader, best_rmse, stop_cnt=0, k=k, valid=False)
    print(f" TEST RMSE : {rmse} | TEST MAE : {mae} | NDCG@10 : {neg_ndcg} | Precision@10 : {neg_precision}")


def test(model, device, data_loader, best_rmse, stop_cnt, k=10, valid=True):

    if not valid:
        model.load_state_dict(torch.load(checkpoint))
        
    model.eval()
    metrics = Metrics()
    rmse = 0.0
    u_lst, p_lst, tmp_pred, targets, masks = [], [], [], [], []
    tqdm_iterator = tqdm(data_loader,desc="Testing (X / X Steps) (loss=X.X)",bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
    for step, batch in tqdm(enumerate(tqdm_iterator)):
        batch = {k:v.to(device) for k,v in batch.items()}
        r_target = batch['rating']
        mask = (r_target!=0).detach().cpu().tolist()
        r_pred, t_pred, reg_term = model(batch)
        r_pred = r_pred.squeeze()
        
        # processing for Rank metric
        pred = list(r_pred.detach().cpu().numpy())
        gt = list(r_target.detach().cpu().numpy())
        users = list(batch['user_id'].detach().cpu().numpy())
        items = list(batch['product_id'].detach().cpu().numpy())
        
        u_lst.extend(users)
        p_lst.extend(items)
        tmp_pred.extend(pred)
        targets.extend(gt)      
        masks.extend(mask)
        # RMSE
        loss = metrics.RMSE(r_pred, r_target)
        tqdm_iterator.set_description(
                        "Testing (%d / %d Steps) (loss=%2.5f)" % (step, len(tqdm_iterator), loss.item()/(step+1)))
        
    pred_t = torch.tensor(tmp_pred)
    target_t = torch.tensor(targets)
    mask_t = torch.tensor(masks)
    u_t = torch.tensor(u_lst)[mask_t]
    i_t = torch.tensor(p_lst)[mask_t]

    # shared benchmark metrics
    rmse = _rmse(target_t[mask_t], pred_t[mask_t])
    mae = _mae(target_t[mask_t], pred_t[mask_t])
    if not valid:
        dump_predictions("trustsvd", DATASET, SEED, u_t, i_t, target_t[mask_t], pred_t[mask_t])

    # --- [implicit/ranking] NDCG / precision block -- disabled for the explicit-feedback
    # --- benchmark (RMSE/MAE only). Kept commented so it can be switched back on later.
    # user_item_df = pd.DataFrame({'user_id':u_lst, 'product_id':p_lst, 'pred':tmp_pred, 'rating':targets})
    # df = metrics.get_logits(user_item_df)
    # ndcg = 0; cnt = 0
    # neg_n, neg_p, neg_r = [],[],[]
    # for _,group in df.groupby('user_id'):
    #     items = group['product_id'].values.astype(np.int32)
    #     logits = group['logits'].values.astype(np.float64)
    #     ratings = group['rating'].values.astype(np.int32)
    #     ndcg, new_k, precision, recall = metrics.rank_metrics(items, logits, ratings, k)
    #     neg_n.append(ndcg*(new_k/k)); neg_p.append(precision); neg_r.append(recall)
    # neg_ndcg = sum(neg_n)/len(neg_n)
    # neg_precision = sum(neg_p)/len(neg_p)
    neg_ndcg = 0.0
    neg_precision = 0.0

    rmse = round(float(rmse), 4)

    # compare & update
    if valid and (round(best_rmse,4)>round(rmse,4)):
        best_rmse = rmse
        print("Best model saved")
        torch.save(model.state_dict(), checkpoint)
        stop_cnt = 0
    else:
        stop_cnt+=1

    return best_rmse, rmse, mae, stop_cnt, neg_ndcg, neg_precision
    

def main():
    global checkpoint_path, checkpoint, lambda_, lambda_t, DATASET, SEED
    args = get_args()
    DATASET, SEED = args.dataset, args.seed

    # fix seed / reproudcibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    data_path = os.path.join('data', args.dataset)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    checkpoint_path = os.path.join(os.getcwd(), 'checkpoint', args.dataset)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint = os.path.join(checkpoint_path, f'trustsvd_lr_{args.lr}.pt')

    rating_train, rating_valid, rating_test, trust_matrix = load_data(data_path, args.seed, args.neg_train, args.regen)
    rating_total = pd.concat([rating_train, rating_valid, rating_test])
    num_users, num_items = trust_matrix.shape[0], rating_total.product_id.nunique()+1
    print(f"# of Users : {num_users} / # of Items : {num_items}")
    # device
    device = torch.device(f'cuda:{args.id}' if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
    # device = torch.device('cpu')
    print(device)
    # Get bias
    mu = rating_train.loc[rating_train['rating'] > 0, 'rating'].mean().item()
    # Get interaction dictionary
    user_item_dict = rating_train.groupby('user_id')['product_id'].unique().to_dict()
    item_user_dict = rating_train.groupby('product_id')['user_id'].unique().to_dict()
    trust_dok = sparse.dok_matrix(trust_matrix)
    user_user_dict = {}
    user_user_in_dict = {}
    for u in range(trust_matrix.shape[0]):
        user_user_dict[u] = list(map(lambda x:x[1], trust_dok[u].keys()))
        user_user_in_dict[u] = list(map(lambda x:x[0], trust_dok[:, u].keys()))

    trainset = MyDataset(rating_train, user_item_dict, item_user_dict, user_user_dict, user_user_in_dict)
    validset = MyDataset(rating_valid, user_item_dict, item_user_dict, user_user_dict, user_user_in_dict)
    testset = MyDataset(rating_test, user_item_dict, item_user_dict, user_user_dict, user_user_in_dict)
    

    def collate_fn(batch):
        new_batch = {
            'user_id':torch.stack([data['user_id'] for data in batch]),
            'product_id':torch.stack([data['product_id'] for data in batch]),
            'rating':torch.stack([data['rating'] for data in batch]),

            'i_u':pad_sequence([data['i_u'] for data in batch], batch_first=True, padding_value=0),
            't_u':pad_sequence([data['t_u'] for data in batch], batch_first=True, padding_value=0),

            'len_u_j':torch.stack([data['len_u_j'] for data in batch]),
            'len_u_i':pad_sequence([data['len_u_i'] for data in batch], batch_first=True, padding_value=0),
            'len_t_v':pad_sequence([data['len_t_v'] for data in batch], batch_first=True, padding_value=0)
        }
        return new_batch

    if args.dataset=='yelp':
        num_workers=0
    else:
        num_workers=8
    train_loader = DataLoader(trainset, batch_size=args.bs, collate_fn=collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(validset, batch_size=args.bs, collate_fn=collate_fn, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(testset, batch_size=args.bs, collate_fn=collate_fn, num_workers=num_workers, shuffle=False)    
    
    # regularization params
    if args.dataset=='ciao_timestamp':
        lambda_ = 0.5
        lambda_t = 1.0
    elif args.dataset=='epinions':
        lambda_ = 0.8
        lambda_t = 0.5
    else:
        lambda_ = 1.0
        lambda_t = 0.5

    # model
    # model = TrustSVD(num_users, num_items, args.d_model, mu, lambda_, lambda_t, user_item_dict, item_user_dict, user_user_dict, device)
    model = TrustSVD(num_users, num_items, args.d_model, mu, lambda_, lambda_t, device).to(device)
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Train & Valid & Eval
    if args.mode=='train':
        print("Start Training")
        run(args.num_epochs, model, optimizer, device, train_loader, valid_loader, test_loader, trust_matrix)
    else:
        # Eval(test)
        best_rmse, rmse, mae, stop_cnt, neg_ndcg, neg_precision = test(model.to(device), device, test_loader, best_rmse=0, stop_cnt=0, k=args.k, valid=False)
        print(f" TEST RMSE : {rmse} | TEST MAE : {mae} | NDCG@10 : {neg_ndcg} | Precision@10 : {neg_precision}")
    
if __name__=='__main__':
    main()
