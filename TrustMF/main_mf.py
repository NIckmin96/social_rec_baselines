import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_preprocess.metrics import rmse as _rmse, mae as _mae, dump_predictions  # noqa: E402

from data_preprocess import process_data
from utils import *
from TrustMF import *

import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='single')
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default='ciao')
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)      # accelerate plain SGD
    parser.add_argument("--lr_factor", type=float, default=0.9)     # ReduceLROnPlateau
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=5.0)     # <=0 disables
    parser.add_argument("--d_model", type=float, default=10)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--regen", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--neg_train", action='store_true')
    args = parser.parse_args()
    return args

def load_data(data, seed, regen, neg_train):
    train_dir = os.path.join('data', data, f'rating_train_{seed}.csv')
    valid_dir = os.path.join('data', data, f'rating_valid_{seed}.csv')
    test_dir = os.path.join('data', data, f'rating_test_{seed}.csv')
    trust_dir = os.path.join('data', data, f'trustnetwork.csv')
    
    if (os.path.isfile(train_dir)&os.path.isfile(valid_dir)&os.path.isfile(test_dir)&os.path.isfile(trust_dir)) and not regen:
        rating_train = pd.read_csv(train_dir)
        rating_valid = pd.read_csv(valid_dir)
        rating_test = pd.read_csv(test_dir)
        trust = pd.read_csv(trust_dir)
        
    else:
        rating_test, rating_valid, rating_train, trust = process_data(data, seed, neg_train)
        
    return rating_train, rating_valid, rating_test, trust

def train(model, device, rating_loader, edge_loader, optimizer, lam=1e-3, grad_clip=5.0):
    model = model.to(device)
    model.train()
    base = model.module if isinstance(model, torch.nn.DataParallel) else model

    def step_opt():
        # gradient clipping keeps the summed-loss + momentum updates stable
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
        optimizer.step()

    # rating term (Eq. 4/5, sum over Omega): one step per rating batch
    running_loss = 0.0
    it = tqdm(rating_loader, ascii=' =', leave=True)
    for step, batch in enumerate(it):
        u = batch['user_id'].to(device)
        j = batch['product_id'].to(device)
        r_target = (batch['rating'].to(device) / 5).float()
        r_hat = model(u, j)
        loss = torch.sum(torch.pow(r_hat - r_target, 2))
        optimizer.zero_grad()
        loss.backward()
        step_opt()
        running_loss += loss.item()
        it.set_description(f"rating ({step+1}/{len(it)}) loss={running_loss/(step+1):.4f}")

    # A-3: trust term (Eq. 4/5, sum over Psi) iterates the observed edges directly,
    #      so every edge is fitted exactly once per epoch, decoupled from how many
    #      ratings its endpoints have. Every row here is a real edge -> target 1.0.
    running_loss = 0.0
    it = tqdm(edge_loader, ascii=' =', leave=True)
    for step, (i, k) in enumerate(it):
        i = i.to(device)
        k = k.to(device)
        t_hat = base.trust_forward(i, k)
        loss = torch.sum(torch.pow(t_hat - 1.0, 2))
        optimizer.zero_grad()
        loss.backward()
        step_opt()
        running_loss += loss.item()
        it.set_description(f"trust ({step+1}/{len(it)}) loss={running_loss/(step+1):.4f}")

    # A-2: weighted-lambda regulariser (Eq. 4/5) once per epoch, so each user /
    #      item / neighbour is penalised exactly once, not once per rating row.
    optimizer.zero_grad()
    (lam * base.reg_loss()).backward()
    step_opt()

def test(epoch, best_rmse, best_mae, stop_cnt,  model, device, checkpoint, data_loader, k=10, eval=False):
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
            pred = model.predict(batch)
            total_rating.extend(batch['rating'].data.cpu().tolist())
            total_pred.extend(pred.data.cpu().tolist())
        
    rating_pred = torch.tensor(total_pred)
    rating = torch.tensor(total_rating)
    users_t = torch.tensor(total_user_idx)
    items_t = torch.tensor(total_item_idx)
    mask = (rating != 0)
    rating_pred = rating_pred[mask]
    rating = rating[mask]
    users_t = users_t[mask]
    items_t = items_t[mask]
    # shared benchmark metrics
    rmse = _rmse(rating, rating_pred)
    mae = _mae(rating, rating_pred)
        
    if not eval:
        if best_rmse-1e-7 > rmse or best_mae-1e-7 > mae:
            torch.save(model.state_dict(), checkpoint)
            best_rmse = rmse
            best_mae = mae
            print("New Best Model Saved.")
            stop_cnt = 0 
        else:
            stop_cnt += 1
        print(f"Epoch {epoch} | VALID RMSE : {rmse} | VALID MAE : {mae}")
        
    else:
        # shared benchmark metrics + prediction dump (explicit-feedback: RMSE/MAE only)
        dump_predictions("trustmf", DATASET, SEED, users_t, items_t, rating, rating_pred)
        # --- [implicit/ranking] NDCG / precision block -- disabled for this benchmark.
        # --- Kept commented so it can be switched back on later.
        # df = pd.DataFrame({'user_id':total_user_idx, 'product_id':total_item_idx, 'rating':total_rating, 'pred':total_pred})
        # df = metrics.get_logits(df)
        # neg_n, neg_p, neg_r = [],[],[]
        # for _,group in df.groupby('user_id'):
        #     items = group['product_id'].values.astype(np.int32)
        #     logits = group['logits'].values.astype(np.float64)
        #     ratings = group['rating'].values.astype(np.int32)
        #     ndcg, new_k, precision, recall = metrics.rank_metrics(items, logits, ratings, k)
        #     neg_n.append(ndcg*(new_k/k)); neg_p.append(precision); neg_r.append(recall)
        # neg_ndcg = sum(neg_n)/len(neg_n)
        # neg_precision = sum(neg_p)/len(neg_p)
        print(f"TEST RMSE : {rmse:.4f} | TEST MAE : {mae:.4f}")
        
    return best_rmse, best_mae, stop_cnt, rmse

def main():
    global DATASET, SEED
    args = get_args()
    DATASET, SEED = args.dataset, args.seed
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
    rating_train, rating_valid, rating_test, trust = load_data(args.dataset, args.seed, args.regen, args.neg_train)
    
    checkpoint_path = os.path.join('checkpoint',args.dataset)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint = os.path.join(checkpoint_path, f'trustmf_lr_{args.lr}.pt')
    
    rating_total = pd.concat([rating_train, rating_valid, rating_test])
    num_users, num_items = max(rating_total.user_id.max(), trust.user_id_1.max(), trust.user_id_2.max())+1, rating_total.product_id.nunique()
    # rating_users = rating_total.user_id.max()+1
    print(f"# of Users : {num_users} / # of Items : {num_items}")
    # print(f'# of Users in Rating : {rating_users}')

    # trust neighbour dicts -- only used below for the weighted-lambda counts
    # (m_b = out-degree, m_w = in-degree). A-3: trust edges are consumed directly
    # by an edge DataLoader, not attached to rating rows.
    truster = trust.groupby('user_id_1')['user_id_2'].unique().to_dict()
    truster = {i: truster[i] if i in truster else np.array([], dtype=np.int64) for i in range(num_users)}
    trustee = trust.groupby('user_id_2')['user_id_1'].unique().to_dict()
    trustee = {i: trustee[i] if i in trustee else np.array([], dtype=np.int64) for i in range(num_users)}

    # observed trust edges (Psi). truster direction: (i, k) = "i trusts k";
    # trustee direction: (i, k) = "k trusts i" (columns swapped).
    truster_edges = trust[['user_id_1', 'user_id_2']].to_numpy()
    trustee_edges = trust[['user_id_2', 'user_id_1']].to_numpy()

    # device preparation
    if args.device=='cpu':
        device = torch.device('cpu')
    elif args.device=='single':
        device = torch.device(f'cuda:{args.id}' if torch.cuda.is_available() else 'cpu')
    elif args.device=='multi' and torch.cuda.device_count() > 1:
        device_idx = [1,2,3]
        print(f"device_idx : {device_idx}")
        device = torch.device(f'cuda:{device_idx[0]}')
    
    print(device)
    
    # dataset / dataloader
    trainset, validset, testset = MyDataset(rating_train), MyDataset(rating_valid), MyDataset(rating_test)

    num_workers=2
    if args.dataset=='ciao':
        num_workers=2
    else:
        num_workers=1
    train_loader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(validset, batch_size=args.bs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=num_workers)

    # A-3: one DataLoader per trust direction, iterated once per epoch.
    truster_edge_loader = DataLoader(EdgeDataset(truster_edges), batch_size=args.bs*8, shuffle=True, num_workers=num_workers)
    trustee_edge_loader = DataLoader(EdgeDataset(trustee_edges), batch_size=args.bs*8, shuffle=True, num_workers=num_workers)

    # org
    # A-4: nb_i / nv_j in the weighted-lambda regulariser (Eq. 4/5) are counts of
    #      TRAINING ratings only -- valid/test counts must not leak into the objective.
    # A-2: index them by user / item id and pad to the full embedding size so the
    #      per-epoch regulariser can gather the whole parameter tensor safely.
    n_b_cnt = rating_train.groupby('user_id')['product_id'].nunique()
    n_v_cnt = rating_train.groupby('product_id')['user_id'].nunique()
    n_b = torch.zeros(num_users + 2, dtype=torch.long)
    n_v = torch.zeros(num_items + 1, dtype=torch.long)
    n_b[torch.as_tensor(n_b_cnt.index.to_numpy(), dtype=torch.long)] = torch.as_tensor(n_b_cnt.to_numpy(), dtype=torch.long)
    n_v[torch.as_tensor(n_v_cnt.index.to_numpy(), dtype=torch.long)] = torch.as_tensor(n_v_cnt.to_numpy(), dtype=torch.long)
    
    truster_cnt = dict(map(lambda kv: (kv[0], len(kv[1])), truster.items()))
    trustee_cnt = dict(map(lambda kv: (kv[0], len(kv[1])), trustee.items()))

    m_b = torch.zeros(num_users + 2, dtype=torch.long)
    m_w = torch.zeros(num_users + 2, dtype=torch.long)
    for i, v in truster_cnt.items():
        m_b[i] = v
    for i, v in trustee_cnt.items():
        m_w[i] = v
    # m_w = torch.tensor(list(truster_cnt.values()))
    # m_b = torch.tensor(list(trustee_cnt.values())) 
    
    # model/optimizer initialization
    truster_mf = Truster(num_users, num_items, args.d_model, n_b, n_v, m_b, m_w)
    trustee_mf = Trustee(num_users, num_items, args.d_model, n_b, n_v, m_b, m_w)
    # truster_optim = torch.optim.Adam(truster_mf.parameters(), lr=args.lr)
    # trustee_optim = torch.optim.Adam(trustee_mf.parameters(), lr=args.lr)
    truster_optim = torch.optim.SGD(truster_mf.parameters(), lr=args.lr, momentum=args.momentum)
    trustee_optim = torch.optim.SGD(trustee_mf.parameters(), lr=args.lr, momentum=args.momentum)
    # decay the LR when the validation RMSE stops improving
    sched_kw = dict(mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    truster_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(truster_optim, **sched_kw)
    trustee_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(trustee_optim, **sched_kw)

    if args.device=='multi' and torch.cuda.device_count() > 1:
        print("Using Multi-GPU Training")
        truster_mf = nn.DataParallel(truster_mf, device_ids=device_idx)
        trustee_mf = nn.DataParallel(trustee_mf, device_ids=device_idx)

    # train / valid / test
    best_rmse = round(1e9, 3); best_mae = round(1e9, 3); stop_cnt = 0
    if not args.eval:
        for epoch in range(1, args.num_epochs+1):
            if stop_cnt == 30:
                break
            # Keep only one model on GPU at a time to avoid OOM spikes.
            trustee_mf.to('cpu')
            truster_mf.to(device)
            train(truster_mf, device, train_loader, truster_edge_loader, truster_optim, grad_clip=args.grad_clip) # Truster model train
            truster_mf.to('cpu')
            torch.cuda.empty_cache()

            truster_mf.to('cpu')
            trustee_mf.to(device)
            train(trustee_mf, device, train_loader, trustee_edge_loader, trustee_optim, grad_clip=args.grad_clip) # Trustee model train
            trustee_mf.to('cpu')
            torch.cuda.empty_cache()
            model = TrustMF(truster_mf, trustee_mf)
            best_rmse, best_mae, stop_cnt, cur_rmse = test(epoch, best_rmse, best_mae, stop_cnt, model, device, checkpoint, valid_loader, k=args.k, eval=args.eval)
            truster_sched.step(cur_rmse)
            trustee_sched.step(cur_rmse)
            print(f"Epoch {epoch} | lr {truster_optim.param_groups[0]['lr']:.2e} | best RMSE {best_rmse:.4f}")

        model.load_state_dict(torch.load(checkpoint))
        test(epoch=None, best_rmse=best_rmse, best_mae=best_mae, stop_cnt=stop_cnt, model=model, device=device, checkpoint=checkpoint, data_loader=test_loader, k=args.k, eval=True)

    else:
        model = TrustMF(truster_mf, trustee_mf)
        model.load_state_dict(torch.load(checkpoint))
        test(epoch=None, best_rmse=best_rmse, best_mae=best_mae, stop_cnt=stop_cnt, model=model, device=device, checkpoint=checkpoint, data_loader=test_loader, k=args.k, eval=True)

if __name__ == '__main__':
    main()
