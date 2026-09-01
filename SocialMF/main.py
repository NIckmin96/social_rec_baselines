import os
import random
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_preprocess.metrics import rmse as _rmse, mae as _mae, dump_predictions  # noqa: E402

from data_preprocess import process_data
from utils import *
from SocialMF import SocialMF

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regen', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--neg_train', action='store_true')
    # paper section 6.3: best lambda_T is dataset-dependent (Epinions: 5, Flixster: 1)
    parser.add_argument('--lambda_t', type=float, default=5.0)
    # paper: lambda_U = lambda_V = 0.1, K in {5, 10}
    parser.add_argument('--lambda_u', type=float, default=0.1)
    parser.add_argument('--lambda_v', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)      # accelerate plain SGD
    parser.add_argument('--lr_factor', type=float, default=0.9)     # ReduceLROnPlateau
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=5.0)     # <=0 disables

    args = parser.parse_args()
    return args

def collate_fn(batch):
    user_idx, item_idx, rating, rating_raw = zip(*batch)
    user_idx = torch.stack(user_idx, dim=0)
    item_idx = torch.stack(item_idx, dim=0)
    rating = torch.stack(rating, dim=0)
    rating_raw = torch.stack(rating_raw, dim=0)
    return user_idx, item_idx, rating, rating_raw

def collate_fn_user(batch):
    user_idx, neighbor_indices, neighbor_values = zip(*batch)
    user_idx = torch.stack(user_idx, dim=0)
    neighbor_indices = torch.nn.utils.rnn.pad_sequence(
        neighbor_indices, batch_first=True, padding_value=0
    )
    neighbor_values = torch.nn.utils.rnn.pad_sequence(
        neighbor_values, batch_first=True, padding_value=0
    ).float()
    return user_idx, neighbor_indices, neighbor_values

def load_data(data_path, seed, regen, neg_train):
    rating_train_path = os.path.join(data_path, f'rating_train_{seed}.csv')
    rating_valid_path = os.path.join(data_path, f'rating_valid_{seed}.csv')
    rating_test_path = os.path.join(data_path, f'rating_test_{seed}.csv')
    trust_matrix_path = os.path.join(data_path, 'trust_matrix.npz')
    exist = (os.path.isfile(rating_train_path) & os.path.isfile(rating_valid_path) & os.path.isfile(rating_test_path) & os.path.isfile(trust_matrix_path))
    if exist and (not regen):
        rating_train = pd.read_csv(rating_train_path)
        rating_valid = pd.read_csv(rating_valid_path)
        rating_test = pd.read_csv(rating_test_path)
        trust_matrix = sparse.load_npz(trust_matrix_path)
    else:
        rating_train, rating_valid, rating_test, trust_matrix = process_data(data_path, seed, neg_train)

    return rating_train, rating_valid, rating_test, trust_matrix

def train(model, device, train_loader, user_loader, item_loader, valid_loader, optimizer, scheduler, epochs, checkpoint, grad_clip=5.0):
    model.to(device)
    best_rmse = 9999; stop_cnt = 0

    def step_opt():
        # gradient clipping keeps the summed-loss + momentum updates stable
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    for epoch in range(epochs):
        model.train()

        # reconstruction term (eq. 12, 1/2 * sum over observed ratings Omega):
        # one optimizer step per rating batch. Simple sum -- no batch mean -- so the
        # term matches the paper's objective scale.
        running_loss = 0.0
        tqdm_iterator = tqdm(train_loader, ascii=' =', leave=True)
        for step, batch in enumerate(tqdm_iterator):
            user_idx, item_idx, rating, rating_raw = batch
            user_idx, item_idx, rating = user_idx.to(device), item_idx.to(device), rating.to(device)
            r_hat = model(user_idx, item_idx).squeeze(-1)
            loss = 0.5 * torch.sum(torch.pow(r_hat - rating, 2))
            optimizer.zero_grad()
            loss.backward()
            step_opt()
            running_loss += loss.item()
            tqdm_iterator.set_description(
                f"recon ({step+1}/{len(tqdm_iterator)}) loss={running_loss/(step+1):.4f}")

        # regularization terms (eq. 12): summed over the FULL user / item sets and
        # applied once per epoch (the paper treats the priors as a full-batch term),
        # not once per rating row -- so each user / item is penalised exactly once and
        # the lambda_U / lambda_V / lambda_T weights act at their paper scale.
        #   lambda_U/2 * sum_u ||U_u||^2
        #   lambda_T/2 * sum_u ||U_u - sum_{v in N_u} T_uv U_v||^2
        #   lambda_V/2 * sum_i ||V_i||^2
        optimizer.zero_grad()
        reg_running = 0.0
        for u_user_idx, u_neighbor_indices, u_neighbor_values in user_loader:
            u_user_idx = u_user_idx.to(device)
            u_neighbor_indices = u_neighbor_indices.to(device)
            u_neighbor_values = u_neighbor_values.to(device)
            reg_u, reg_t = model.user_reg_loss(u_user_idx, u_neighbor_indices, u_neighbor_values)
            loss = 0.5 * torch.sum(reg_u) + 0.5 * torch.sum(reg_t)
            loss.backward()
            reg_running += loss.item()
        for u_item_idx in item_loader:
            u_item_idx = u_item_idx.to(device)
            loss = 0.5 * torch.sum(model.item_reg_loss(u_item_idx))
            loss.backward()
            reg_running += loss.item()
        step_opt()
        print(f"[epoch {epoch+1}] recon {running_loss/len(train_loader):.4f} | reg {reg_running:.2f}")

        # valid + LR schedule (decay when validation RMSE plateaus)
        best_rmse, stop_cnt, cur_rmse = valid(model, device, valid_loader, checkpoint, best_rmse, stop_cnt)
        scheduler.step(cur_rmse)
        print(f"Epoch {epoch+1} | lr {optimizer.param_groups[0]['lr']:.2e} | best RMSE {best_rmse:.4f}")
        if stop_cnt==30:
            break

def valid(model, device, valid_loader, checkpoint, best_rmse, stop_cnt):
    model.eval()
    total_user_idx, total_item_idx,  = [],[]
    total_rating, total_pred, total_rating_raw = [],[],[]

    with torch.no_grad():
        for batch in valid_loader:
            user_idx, item_idx, rating, rating_raw = batch
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device)
            rating_raw = rating_raw.to(device)
            r_hat = model(user_idx, item_idx).squeeze()
            total_user_idx.extend(user_idx.data.cpu().tolist())
            total_item_idx.extend(item_idx.data.cpu().tolist())
            total_rating.extend(rating.data.cpu().tolist())
            total_pred.extend(r_hat.data.cpu().tolist())
            total_rating_raw.extend(rating_raw.data.cpu().tolist())

    rating_pred = torch.tensor(total_pred).to(device)
    rating = torch.tensor(total_rating).to(device)
    rating_raw = torch.tensor(total_rating_raw).to(device)
    mask = (rating_raw > 0).to(device)
    rating_pred = rating_pred[mask]
    rating = rating[mask]
    assert (rating_raw[mask] == 0).sum().item() == 0
    # scaling to 1~5
    rating = rating*4.0 + 1.0
    rating_pred = rating_pred*4.0 + 1.0
    # RMSE / MAE
    rmse = torch.sqrt(F.mse_loss(rating_pred, rating, reduction='mean')).item()
    mae = F.l1_loss(rating_pred, rating, reduction='mean').item()

    # Compare at full precision. With a well-scaled init the model reaches its plateau
    # within the first epoch and later gains land past the 4th decimal; rounding to 4 dp
    # here made `best_rmse > rmse` false for every real-but-tiny improvement, so the
    # checkpoint got stuck at epoch 1 while stop_cnt still climbed to an early stop.
    prev_best = best_rmse
    if rmse < prev_best:
        torch.save(model.state_dict(), checkpoint)
        best_rmse = rmse
        print(f"New Best Model Saved. (RMSE {rmse:.6f})")
    # only a >1e-5 gain counts as progress for early-stopping, so a true plateau still stops
    if rmse < prev_best - 1e-7:
        stop_cnt = 0
    else:
        stop_cnt += 1
    print(f"VALID RMSE : {rmse:.4f} | VALID MAE : {mae:.4f} | BEST RMSE : {best_rmse:.4f}")

    return best_rmse, stop_cnt, rmse

def test(model, device, loader, k=10):
    model.to(device)
    model.eval()

    metrics = Metrics()
    total_user_idx, total_item_idx,  = [],[]
    total_rating, total_pred, total_rating_raw = [],[],[]

    with torch.no_grad():
        for batch in loader:
            user_idx, item_idx, rating, rating_raw = batch
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device)
            rating_raw = rating_raw.to(device)
            r_hat = model(user_idx, item_idx).squeeze()
            total_user_idx.extend(user_idx.data.cpu().tolist())
            total_item_idx.extend(item_idx.data.cpu().tolist())
            total_rating.extend(rating.data.cpu().tolist())
            total_pred.extend(r_hat.data.cpu().tolist())
            total_rating_raw.extend(rating_raw.data.cpu().tolist())

    rating_pred = torch.tensor(total_pred)
    rating = torch.tensor(total_rating)
    rating_raw = torch.tensor(total_rating_raw)
    users_t = torch.tensor(total_user_idx)
    items_t = torch.tensor(total_item_idx)
    mask = (rating_raw > 0)
    rating_pred = rating_pred[mask]
    rating = rating[mask]
    users_t = users_t[mask]
    items_t = items_t[mask]
    assert (rating_raw[mask] == 0).sum().item() == 0
    # scaling to 1~5
    rating = rating*4.0 + 1.0
    rating_pred = rating_pred*4.0 + 1.0

    # shared benchmark metrics + prediction dump
    _, _m = dump_predictions("socialmf", DATASET, SEED, users_t, items_t, rating, rating_pred)
    rmse = _m["rmse"]
    mae = _m["mae"]

    # # rank metric
    # df = pd.DataFrame({'user_id':total_user_idx, 'product_id':total_item_idx, 'rating':total_rating, 'pred':total_pred})
    # # print(df['pred'])
    # df = metrics.get_logits(df)
    # neg_n, neg_p, neg_r = [],[],[]
    # p_lst, r_lst, n_lst = [],[],[]
    # filtered_n, filtered_p, filtered_r = [],[],[]
    # for _,group in df.groupby('user_id'):
    #     items = group['product_id'].values.astype(np.int32)
    #     logits = group['logits'].values.astype(np.float64)
    #     ratings = group['rating'].values.astype(np.int32)
    #     # negative sample 포함된 metrics
    #     ndcg, new_k, precision, recall = metrics.rank_metrics(items, logits, ratings, k)
    #     neg_n.append(ndcg*(new_k/k))
    #     neg_p.append(precision)
    #     neg_r.append(recall)
    #     negative sample 제외된 metrics
    #     mask = (ratings!=0)
    #     ndcg, new_k, precision, recall = metrics.rank_metrics(items[mask], logits[mask], ratings[mask], k)
    #     if len(items[mask])<10:
    #         filtered_n.append(ndcg*(new_k/k))
    #         filtered_p.append(precision)
    #         filtered_r.append(recall)
    #         continue
    #     n_lst.append(ndcg*(new_k/k))
    #     p_lst.append(precision)
    #     r_lst.append(recall)

    # negative sample이 포함된 metrics
    # neg_ndcg = sum(neg_n)/len(neg_n)
    # neg_precision = sum(neg_p)/len(neg_p)
    # neg_recall = sum(neg_r)/len(neg_r)
    # # negative sample이 제외된 전체 user에 대한 metrics
    # total_ndcg = (sum(n_lst)+sum(filtered_n))/(len(n_lst)+len(filtered_n))
    # total_precision = (sum(p_lst)+sum(filtered_p))/(len(p_lst)+len(filtered_p))
    # total_recall = (sum(r_lst)+sum(filtered_r))/(len(r_lst)+len(filtered_r))
    # # negative sample이 제외 + item개수가 10개 이상인 user에 대한 metric
    # filtered_ndcg = sum(n_lst)/len(n_lst)
    # filtered_precision = sum(p_lst)/len(p_lst)
    # filtered_recall = sum(r_lst)/len(r_lst)
    print(f"TEST RMSE : {rmse:.4f} | TEST MAE : {mae:.4f}")

def main():
    global SEED, DATASET
    args = get_args()
    SEED = args.seed
    DATASET = args.dataset

    # fix seed / reproducibility -- without this, model init (.normal_) and the
    # shuffled train/user/item DataLoaders draw from an unseeded global RNG, so the
    # same command gives a different result every run.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    data_path = os.path.join('data', args.dataset)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    rating_train, rating_valid, rating_test, trust_matrix = load_data(data_path, args.seed, args.regen, args.neg_train)
    rating = pd.concat([rating_train, rating_valid, rating_test])
    # num_users must come from the trust matrix, not from rating['user_id'].nunique():
    # the trust network includes users with no ratings (needed for trust propagation),
    # so it can be larger than the set of users who ever appear in a rating row.
    num_users = trust_matrix.shape[0] - 1
    num_items = rating.product_id.nunique()
    # Dataset & DataLoader
    trainset, validset, testset = MyDataset(rating_train), MyDataset(rating_valid), MyDataset(rating_test)
    train_loader = DataLoader(trainset, batch_size=512, num_workers=3, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(validset, batch_size=512, num_workers=3, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(testset, batch_size=512, num_workers=3, shuffle=False, collate_fn=collate_fn)
    userset = UserGraphDataset(num_users, trust_matrix)
    user_loader = DataLoader(userset, batch_size=512, num_workers=3, shuffle=True, collate_fn=collate_fn_user)
    itemset = ItemDataset(num_items)
    item_loader = DataLoader(itemset, batch_size=512, num_workers=3, shuffle=True)
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # SocialMF / checkpoint
    model = SocialMF(num_users, num_items, args.d_model,
                     lambda_u=args.lambda_u, lambda_v=args.lambda_v, lambda_t=args.lambda_t)
    checkpoint = os.path.join('checkpoint', args.dataset)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    checkpoint = os.path.join(checkpoint, f'model_{args.seed}.pkt')
    # optimizer + LR scheduler (decay on validation-RMSE plateau, like TrustMF)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    if not args.eval:
        # train / valid
        train(model, device, train_loader, user_loader, item_loader, valid_loader,
              optimizer, scheduler, args.num_epochs, checkpoint, grad_clip=args.grad_clip)
    # test
    model.load_state_dict(torch.load(checkpoint))
    test(model, device, test_loader, k=args.k)

if __name__ == '__main__':
    main()
