import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os

from data_preprocess import process_data
"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""
def apply_softmax(group):
    logits = torch.tensor(group['pred'].values, dtype=torch.float)
    logits = F.softmax(logits, dim=0)
    group['logits'] = logits
    return group

def get_logits(df:pd.DataFrame):
    df = df.groupby('user_id').apply(apply_softmax).reset_index(drop=True)
    return df

def rank_metrics(items, logits, ratings, k=10):
        eps = 1e-10
        new_k = min(k, len(ratings))
        gt_items = items[ratings!=0] # 실제로 interact한 items
        _, rec_indices = torch.from_numpy(logits).topk(new_k)
        recommended_i = torch.from_numpy(items)[rec_indices].flatten()
        recommended_r = torch.from_numpy(ratings)[rec_indices].flatten()
        
        _, ideal_indices = torch.from_numpy(ratings).topk(new_k)
        ideal_i = torch.from_numpy(items)[ideal_indices].flatten()
        ideal_r = torch.from_numpy(ratings)[ideal_indices].flatten()
        discount = torch.log2(torch.arange(new_k)+2)
        item_mask = torch.tensor(list(map(lambda x:1 if x in set(gt_items.tolist()) else 0, list(recommended_i))))
        
        # DCG = torch.sum(recommended_r*item_mask/discount)
        DCG = torch.sum(recommended_r/discount)
        IDCG = torch.sum(ideal_r/discount)
        NDCG = DCG/(IDCG+eps)
        # NDCG *= (new_k/k) # 보정
        assert NDCG<=1.0
        
        # precision
        TP = set(recommended_i.tolist()).intersection(set(gt_items.tolist()))
        precision = round(len(TP)/new_k, 4) if new_k>0 else 0.0
        recall = round(len(TP)/len(gt_items), 4) if len(gt_items)>0 else 0.0
        
        return NDCG, new_k, precision, recall

class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0

def test(model, device, test_loader):
    model.eval()
    u_lst, p_lst, tmp_pred, target, masks = [], [], [], [], []
    user_item_df = pd.DataFrame(columns=['user_id','product_id','pred','rating'])
    rmse = 0.0; cnt=0
    test_iterator = tqdm(test_loader, ascii=' =', dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for step, (test_u, test_v, tmp_target) in enumerate(test_iterator):
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            mask = (tmp_target!=0)
            val_output = model.forward(test_u, test_v)
            
            pred = list(val_output.data.cpu().numpy())
            gt = list(tmp_target.data.cpu().numpy())
            users = list(test_u.data.cpu().numpy())
            items = list(test_v.data.cpu().numpy())

            u_lst.extend(users)
            p_lst.extend(items)
            tmp_pred.extend(pred)
            target.extend(gt)
            masks.extend(mask.data.cpu().numpy())

            if val_output[mask].numel()==0:
                continue

            rmse=torch.sqrt(F.mse_loss(val_output[mask], tmp_target[mask])).item()          
            test_iterator.set_description(
                        "Testing (%d / %d Steps) (loss=%2.5f)" % (step, len(test_iterator), rmse))
            
    pred_t = torch.tensor(tmp_pred)
    target_t = torch.tensor(target)
    mask_t = torch.tensor(masks)
        
    rmse = torch.sqrt(F.mse_loss(pred_t[mask_t], target_t[mask_t], reduction='mean'))
    mae = F.l1_loss(pred_t[mask_t], target_t[mask_t], reduction='mean')

    user_item_df = pd.DataFrame({'user_id':u_lst, 'product_id':p_lst, 'pred':tmp_pred, 'rating':target})
    df = get_logits(user_item_df)
    
    k=10
    neg_n, neg_p, neg_r = [],[],[]
    p_lst, r_lst, n_lst = [],[],[]
    filtered_n, filtered_p, filtered_r = [],[],[]
    for _,group in df.groupby('user_id'):
        items = group['product_id'].values.astype(np.int32)
        logits = group['logits'].values.astype(np.float64)
        ratings = group['rating'].values.astype(np.int32)
        # negative sample 포함된 metrics
        ndcg, new_k, precision, recall = rank_metrics(items, logits, ratings, k) 
        neg_n.append(ndcg)
        neg_p.append(precision)
        neg_r.append(recall)
        # negative sample 제외된 metrics
        mask = (ratings!=0)
        ndcg, new_k, precision, recall = rank_metrics(items[mask], logits[mask], ratings[mask], k)
        if len(items[mask])<10:
            filtered_n.append(ndcg)
            filtered_p.append(precision)
            filtered_r.append(recall)
            continue
        n_lst.append(ndcg)
        p_lst.append(precision)
        r_lst.append(recall)
    
    # negative sample이 포함된 metrics    
    neg_ndcg = sum(neg_n)/len(neg_n)
    neg_precision = sum(neg_p)/len(neg_p)
    neg_recall = sum(neg_r)/len(neg_r)
    # # negative sample이 제외된 전체 user에 대한 metrics
    # total_ndcg = (sum(n_lst)+sum(filtered_n))/(len(n_lst)+len(filtered_n))
    # total_precision = (sum(p_lst)+sum(filtered_p))/(len(p_lst)+len(filtered_p))
    # total_recall = (sum(r_lst)+sum(filtered_r))/(len(r_lst)+len(filtered_r))
    # # negative sample이 제외 + item개수가 10개 이상인 user에 대한 metric
    # filtered_ndcg = sum(n_lst)/len(n_lst)
    # filtered_precision = sum(p_lst)/len(p_lst)
    # filtered_recall = sum(r_lst)/len(r_lst)
    
    return rmse, mae, neg_ndcg, neg_precision, neg_recall


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regen', type=bool, default=False)
    parser.add_argument('--neg_train', type=bool, default=False)
    parser.add_argument('--multi_hops', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    path_data = os.path.join('data', args.dataset, f'dataset_{args.seed}.pickle')
    if os.path.isfile(path_data) and not args.regen:
        data_file = open(path_data, 'rb')
        history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, valid_u, valid_v, valid_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(data_file)
    else:
        data_file = process_data(args.dataset, args.seed, args.neg_train, args.multi_hops)
        history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, valid_u, valid_v, valid_r, test_u, test_v, test_r, social_adj_lists, ratings_list = data_file
    
    
    checkpoint = os.path.join('checkpoint', args.dataset)
    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)
    
    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)
    
    # please add the validation set
    
    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                             torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    # num_users = history_u_lists.__len__()
    # num_items = history_v_lists.__len__()
    num_users = max(max(train_u), max(valid_u), max(test_u))+1
    num_items = max(max(train_v), max(valid_v), max(test_v))+1
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users+1, embed_dim).to(device) # empty history를 위해서 +1
    v2e = nn.Embedding(num_items+1, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings+1, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    
    if not args.eval:
        for epoch in range(1, args.epochs + 1):

            train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
            val_rmse, val_mae, val_ndcg, val_precision, val_recall = test(graphrec, device, valid_loader)
            # please add the validation set to tune the hyper-parameters based on your datasets.

            # early stopping (no validation set in toy dataset)
            if best_rmse > val_rmse:
                best_rmse = val_rmse
                best_mae = val_mae
                endure_count = 0
                # save model
                if not os.path.exists(checkpoint):
                    os.makedirs(checkpoint)
                torch.save(graphrec.state_dict(), os.path.join(checkpoint, f'model_{args.seed}.pkt'))
            else:
                endure_count += 1
            print("rmse: %.4f, mae:%.4f " % (val_rmse, val_mae))

            if endure_count > 5:
                break
        
    # test
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    graphrec.load_state_dict(torch.load(os.path.join(checkpoint, f'model_{args.seed}.pkt')))
    test_rmse, test_mae, test_ndcg, test_precision, test_recall = test(graphrec, device, test_loader)
    print(f"TEST RMSE : {test_rmse:.4f} | TEST MAE : {test_mae:.4f}\n\
        NEG NDCG@{10} : {test_ndcg:.4f} | NEG PRECISION@{10} : {test_precision:.4f}")


if __name__ == "__main__":
    main()
