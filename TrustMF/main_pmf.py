import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_preprocess import process_data
from utils import *
from TrustPMF import *

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='single')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default='ciao')
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=float, default=10)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--regen", type=bool, default=False)
    args = parser.parse_args()
    return args

def load_data(data, regen=False):
    train_dir = os.path.join('data', data, 'rating_train.csv')
    valid_dir = os.path.join('data', data, 'rating_valid.csv')
    test_dir = os.path.join('data', data, 'rating_test.csv')
    trust_dir = os.path.join('data', data, 'trustnetwork.csv')
    
    if (os.path.isfile(train_dir)&os.path.isfile(valid_dir)&os.path.isfile(test_dir)&os.path.isfile(trust_dir)) and not regen:
        rating_train = pd.read_csv(train_dir)
        rating_valid = pd.read_csv(valid_dir)
        rating_test = pd.read_csv(test_dir)
        trust = pd.read_csv(trust_dir)
        
    else:
        rating_test, rating_valid, rating_train, trust = process_data(data)
        
    return rating_train, rating_valid, rating_test, trust

def train(model, device, data_loader, optimizer):
    model = model.to(device)
    model.train()
    running_loss = 0.0
    tqdm_iterator = tqdm(data_loader, ascii=' =', leave=True)
    for step, batch in enumerate(tqdm_iterator):
        r_target = (batch['rating']/5).float().unsqueeze(-1)
        t_truster = torch.where(batch['truster']==model.num_users+1, 0, 1).float()
        t_trustee = torch.where(batch['trustee']==model.num_users+1, 0, 1).float()
            
        # to device
        batch = {k:v.to(device) for k,v in batch.items()}
        
        r_hat, t_hat_truster, t_hat_trustee, reg_term = model(batch)
        r_loss = torch.sum(torch.pow(r_hat.cpu()-r_target, 2))
        t_loss = (torch.sum(torch.pow(t_hat_truster.cpu()-t_truster, 2)) + torch.sum(torch.pow(t_hat_trustee.cpu()-t_trustee, 2)))
        loss = r_loss*0.5 + t_loss*0.5 + 0.001*reg_term
        
        # backpropagation / optimizer
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        running_loss /= (step+1)
        tqdm_iterator.set_description(
            f"Training ({step+1}/{len(tqdm_iterator)} Steps) (loss={running_loss})"
        )

def test(epoch, best_rmse, stop_cnt,  model, device, checkpoint, data_loader, k=10, eval=False):
    if eval and os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    
    model = model.to(device)
    model.eval()
    
    metrics = Metrics()
    total_user_idx, total_item_idx,  = [],[]
    total_rating, total_pred = [],[]
    tqdm_iterator = tqdm(data_loader, desc="Testing", ascii=' =', leave=True)
    with torch.no_grad():
        for step, batch in enumerate(tqdm_iterator):
            tqdm_iterator.set_description(f"({step+1}/{len(data_loader)} Steps)")
            user_idx, item_idx = batch['user_id'], batch['product_id']
            total_user_idx.extend(user_idx.data.cpu().tolist())
            total_item_idx.extend(item_idx.data.cpu().tolist())
            
            batch = {k:v.to(device) for k,v in batch.items()}
            pred,_,_,_ = model(batch)
            pred*=5
            total_rating.extend(batch['rating'].data.cpu().tolist())
            total_pred.extend(pred.data.cpu().tolist())
        
    rating_pred = torch.tensor(total_pred).to(device)
    rating = torch.tensor(total_rating).to(device)
    mask = (rating!=0).to(device)
    rating_pred = rating_pred*mask
    squared_error = torch.sum(torch.pow(rating_pred-rating, 2))
    mse = squared_error/torch.sum(mask)
    rmse = round(torch.sqrt(mse).item(), 3)
    
    if not eval:
        if best_rmse > rmse:
            torch.save(model.state_dict(), checkpoint)
            best_rmse = rmse
            print("New Best Model Saved.")
            stop_cnt = 0 
        else:
            stop_cnt += 1
        print(f"Epoch {epoch} | VALID RMSE : {rmse}")
        
    else:
        # rank metric
        ndcg=0
        df = pd.DataFrame({'user_id':total_user_idx, 'product_id':total_item_idx, 'rating':total_rating, 'pred':total_pred})
        df = metrics.get_logits(df)
        for i,group in df.groupby('user_id'):
            items = group['product_id'].values.astype(np.int32)
            logits = group['logits'].values.astype(np.float64)
            ratings = group['rating'].values.astype(np.int32)
            _value, _len = metrics.NDCG(items, logits, ratings, k)
            ndcg+=(_value*(_len/k))

        ndcg /= (i+1)
        print(f"TEST RMSE : {rmse} | TEST NDCG@{k} : {ndcg}")    
        
    
    return  best_rmse, stop_cnt

def main():
    args = get_args()
    # fix seed / reproudcibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    # data preparation
    data_path = os.path.join('data', args.dataset)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    rating_train, rating_valid, rating_test, trust = load_data(args.dataset, args.regen)
    
    checkpoint_path = os.path.join('checkpoint',args.dataset)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint = os.path.join(checkpoint_path, f'trustmf_lr_{args.lr}.pt')
    
    rating_total = pd.concat([rating_train, rating_valid, rating_test])
    num_users, num_items = max(rating_total.user_id.max(), trust.user_id_1.max(), trust.user_id_2.max())+1, rating_total.product_id.nunique()
    print(f"# of Users : {num_users} / # of Items : {num_items}")

    # total df
    truster = trust.groupby('user_id_1')['user_id_2'].unique().to_dict()
    truster = {i:truster[i] if i in truster.keys() else np.array([0]) for i in range(num_users)}
    trustee = trust.groupby('user_id_2')['user_id_1'].unique().to_dict()
    trustee = {i:trustee[i] if i in trustee.keys() else np.array([0]) for i in range(num_users)}
    
    rating_train.loc[:, 'truster'] = rating_train['user_id'].apply(lambda x:truster[x] if x in truster else np.array([num_users+1]))
    rating_train.loc[:, 'trustee'] = rating_train['user_id'].apply(lambda x:trustee[x] if x in trustee else np.array([num_users+1]))
    
    rating_valid.loc[:, 'truster'] = rating_valid['user_id'].apply(lambda x:truster[x] if x in truster else np.array([num_users+1]))
    rating_valid.loc[:, 'trustee'] = rating_valid['user_id'].apply(lambda x:trustee[x] if x in trustee else np.array([num_users+1]))
    
    rating_test.loc[:, 'truster'] = rating_test['user_id'].apply(lambda x:truster[x] if x in truster else np.array([num_users+1]))
    rating_test.loc[:, 'trustee'] = rating_test['user_id'].apply(lambda x:trustee[x] if x in trustee else np.array([num_users+1]))
    
    # device preparation
    if args.device=='cpu':
        device = torch.device('cpu')
    elif args.device=='single':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    elif args.device=='multi' and torch.cuda.device_count() > 1:
        device_idx = [1,2,3]
        print(f"device_idx : {device_idx}")
        device = torch.device(f'cuda:{device_idx[0]}')
    
    print(device)
    
    # dataset / dataloader
    trainset, validset, testset = MyDataset(rating_train), MyDataset(rating_valid), MyDataset(rating_test)
    def collate_fn(batch):
        new_batch = {
            'user_id':torch.stack([data['user_id'] for data in batch]),
            'product_id':torch.stack([data['product_id'] for data in batch]),
            'rating':torch.stack([data['rating'] for data in batch]),
            'truster':pad_sequence([data['truster'] for data in batch], batch_first=True, padding_value=num_users+1),
            'trustee':pad_sequence([data['trustee'] for data in batch], batch_first=True, padding_value=num_users+1)
        }
        return new_batch
    
    train_loader = DataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate_fn)
    valid_loader = DataLoader(validset, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate_fn) 
    
    # model/optimizer initialization
    trust_pmf = TrustPMF(num_users, num_items, args.d_model)
    optimizer = torch.optim.Adam(trust_pmf.parameters(), lr=args.lr)
    
    if args.device=='multi' and torch.cuda.device_count() > 1:
        print("Using Multi-GPU Training")
        trust_pmf = nn.DataParallel(trust_pmf, device_ids=device_idx)
    
    # train / valid / test
    best_rmse = round(1e9, 3); stop_cnt = 0
    if not args.eval:
        for epoch in range(1, args.num_epochs+1):
            if stop_cnt > 7:
                break
            train(trust_pmf, device, train_loader, optimizer) # Truster model train
            best_rmse, stop_cnt = test(epoch, best_rmse, stop_cnt, trust_pmf, device, checkpoint, valid_loader, k=args.k, eval=args.eval)
        test(epoch=None, best_rmse=best_rmse, stop_cnt=stop_cnt, model=trust_pmf, device=device, checkpoint=checkpoint, data_loader=test_loader, k=args.k, eval=True)

    else:
        model = TrustPMF(num_users, num_items, args.d_model)
        test(epoch=None, best_rmse=best_rmse, stop_cnt=stop_cnt, model=model, device=device, checkpoint=checkpoint, data_loader=test_loader, k=args.k, eval=True)

if __name__ == '__main__':
    main()