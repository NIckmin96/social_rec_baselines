#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 01 Apr, 2020

@author: chenjiajia
"""

import os
import time
import argparse
import pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from utils import collate_fn
from model import GDSRec
from dataloader import GRDataset

import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_preprocess.metrics import rmse as _rmse, mae as _mae, dump_predictions  # noqa: E402

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', default='datasets/Ciao/', help='dataset directory path: datasets/Ciao/Epinions')
parser.add_argument('--data', default='Ciao', help='corresponding to datapath')
parser.add_argument('--dataset', default=None, help='common dataset name for the prediction dump')
parser.add_argument('--seed', type=int, default=42, help='seed / common split id for the prediction dump')
parser.add_argument('--sigma', type=str, default='0', help='social strength definition')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=256, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=10, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', default=True, help='test')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main():
    print('Loading data...')
    with open(args.dataset_path + 'dataset_'+ args.sigma +'.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)

    with open(args.dataset_path + 'list_'+ args.sigma +'.pkl', 'rb') as f:
        u_items_divlist = pickle.load(f)
        u_items_list = pickle.load(f)
        u_avg_list = pickle.load(f)
        u_users_similar = pickle.load(f)
        u_users_items_list = pickle.load(f)
        u_users_items_divlist = pickle.load(f)
        i_avg_list = pickle.load(f)
        i_users_list = pickle.load(f)
        i_users_divlist = pickle.load(f)
        (user_count, item_count, rate_count) = pickle.load(f)

    test_data = GRDataset(test_set, u_items_divlist, u_avg_list, u_users_similar, u_users_items_divlist,
                          i_users_divlist, i_avg_list)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GDSRec(user_count+1, item_count+1, rate_count+1, args.embed_dim).to(device)

    if args.test:
        print('Load checkpoint and testing...')
        ckpt = torch.load(args.data+'/best_checkpoint_'+args.sigma+'.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        mae, rmse = validate(test_loader, model,
                             dump=(args.dataset, args.seed) if args.dataset else None)
        print("Test: MAE: {:.4f}, RMSE: {:.4f}".format(mae, rmse))
        return


def validate(valid_loader, model, dump=None):
    model.eval()
    u_all, i_all, y_true, y_pred = [], [], [], []
    with torch.no_grad():
        for uids, iids, labels, u_itemsdiv, u_avg, u_users, u_users_items, i_users, i_avg in tqdm(
                valid_loader):
            uids = uids.to(device)
            iids = iids.to(device)
            labels = labels.to(device)
            u_itemsdiv = u_itemsdiv.to(device)
            # u_items = u_items.to(device)
            u_avg = u_avg.to(device)
            u_users = u_users.to(device)
            u_users_items = u_users_items.to(device)
            i_users = i_users.to(device)
            i_avg = i_avg.to(device)

            preds = model(uids, iids, u_itemsdiv, u_avg, u_users, u_users_items, i_users, i_avg).to(device)
            u_all.extend(uids.data.cpu().numpy().tolist())
            i_all.extend(iids.data.cpu().numpy().tolist())
            y_true.extend(labels.data.cpu().numpy().tolist())
            y_pred.extend(preds.squeeze(1).data.cpu().numpy().tolist())

    if dump is not None:
        ds, seed = dump
        dump_predictions("gdsrec", ds, seed, u_all, i_all, y_true, y_pred)
    return _mae(y_true, y_pred), _rmse(y_true, y_pred)


if __name__ == '__main__':
    main()
