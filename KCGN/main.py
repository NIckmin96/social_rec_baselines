# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from ToolScripts.TimeLogger import log
import pickle
import os
import sys
import gc
import random
import argparse
import scipy.sparse as sp
from ToolScripts.utils import loadData
from ToolScripts.utils import load
from ToolScripts.utils import buildSubGraph
from ToolScripts.utils import sparse_mx_to_torch_sparse_tensor
from ToolScripts.utils import mkdir
from dgl import DGLGraph
import dgl
from MyGCN import MODEL
from BPRData import BPRData
import torch.utils.data as dataloader
from DGI.dgi import DGI
import datetime
import time
import networkx as nx
from collections import defaultdict

from dataProcess import main as dp_main
from data_preprocess import process_kcgn


device_gpu = t.device("cuda")
modelUTCStr = datetime.datetime.now().strftime('%m%d%H%M')
# modelUTCStr = str(int(time.time()))[4:]

isLoadModel = False
LOAD_MODEL_PATH = ""

class Model():

    def __init__(self, args, isLoad=False):
        # NDCG 수정 위한 데이터
        with open(os.path.join(os.getcwd(), 'dataset', args.dataset, 'ratings.pkl'), 'rb') as f:
            self.rating = pickle.load(f)
        
        self.args = args
        self.datasetDir = os.path.join(os.getcwd(), "dataset", args.dataset)
        trainMat, validData, multi_adj_time, uuMat, iiMat = self.getData(args)
        self.userNum, self.itemNum = trainMat.shape
        log("uu num = %d"%(uuMat.nnz))
        log("ii num = %d"%(iiMat.nnz))
        self.trainMat = trainMat


        # self.uu_graph = DGLGraph(uuMat)
        uuMat_edge_src, uuMat_edge_dst = uuMat.nonzero()
        self.uu_graph = dgl.graph(data=(uuMat_edge_src, uuMat_edge_dst),
                            idtype=t.int32,
                            num_nodes=uuMat.shape[0],
                            device=device_gpu)
        # self.ii_graph = DGLGraph(iiMat)
        iiMat_edge_src, iiMat_edge_dst = iiMat.nonzero()
        self.ii_graph = dgl.graph(data=(iiMat_edge_src, iiMat_edge_dst),
                            idtype=t.int32,
                            num_nodes=iiMat.shape[0],
                            device=device_gpu)

        #get sub graph message
        uu_subGraph_data = self.datasetDir + f'/uuMat_subGraph_data_{args.seed}.pkl'
        if self.args.clear:
            if os.path.exists(uu_subGraph_data):
                log("clear uu sub graph message")
                os.remove(uu_subGraph_data)

        if os.path.exists(uu_subGraph_data):
            data = load(uu_subGraph_data)
            self.uu_node_subGraph, self.uu_subGraph_adj, self.uu_dgi_node = data
        else:
            log("rebuild uu sub graph message")
            _, self.uu_node_subGraph, self.uu_subGraph_adj, self.uu_dgi_node = buildSubGraph(uuMat, self.args.subNode)
            data = (self.uu_node_subGraph, self.uu_subGraph_adj, self.uu_dgi_node)
            with open(uu_subGraph_data, 'wb') as fs:
                pickle.dump(data, fs)
        
        ii_subGraph_data = self.datasetDir + f'/iiMat_subGraph_data_{args.seed}.pkl'

        if self.args.clear:
            if os.path.exists(ii_subGraph_data):
                log("clear ii sub graph message")
                os.remove(ii_subGraph_data)

        if os.path.exists(ii_subGraph_data):
            data = load(ii_subGraph_data)
            self.ii_node_subGraph, self.ii_subGraph_adj, self.ii_dgi_node = data
        else:
            log("rebuild ii sub graph message")
            _, self.ii_node_subGraph, self.ii_subGraph_adj, self.ii_dgi_node = buildSubGraph(iiMat, self.args.subNode)
            data = (self.ii_node_subGraph, self.ii_subGraph_adj, self.ii_dgi_node)
            with open(ii_subGraph_data, 'wb') as fs:
                pickle.dump(data, fs)

        self.uu_subGraph_adj_tensor = sparse_mx_to_torch_sparse_tensor(self.uu_subGraph_adj).cuda()
        self.uu_subGraph_adj_norm = t.from_numpy(np.sum(self.uu_subGraph_adj, axis=1)).float().cuda()

        self.ii_subGraph_adj_tensor = sparse_mx_to_torch_sparse_tensor(self.ii_subGraph_adj).cuda()
        self.ii_subGraph_adj_norm = t.from_numpy(np.sum(self.ii_subGraph_adj, axis=1)).float().cuda()

        self.uu_dgi_node_mask = np.zeros(self.userNum)
        self.uu_dgi_node_mask[self.uu_dgi_node] = 1
        self.uu_dgi_node_mask = t.from_numpy(self.uu_dgi_node_mask).float().cuda()

        self.ii_dgi_node_mask = np.zeros(self.itemNum)
        self.ii_dgi_node_mask[self.ii_dgi_node] = 1
        self.ii_dgi_node_mask = t.from_numpy(self.ii_dgi_node_mask).float().cuda()
        
        #norm time value
        log("time process")
        self.time_step = self.args.time_step
        log("time step = %.1f hour"%(self.time_step))
        time_step = 3600*self.time_step
        row, col = multi_adj_time.nonzero()
        data = multi_adj_time.data
        minUTC = data.min()
        #data.min = 2
        data = ((data-minUTC)/time_step).astype(np.int64)+2 # timestamp에 대해서 normalization -> multi_adj_time_norm에 저장
        assert np.sum(row == col) == 0
        multi_adj_time_norm = sp.coo_matrix((data, (row, col)), dtype=np.int64, shape=multi_adj_time.shape).tocsr()
        self.maxTime = multi_adj_time_norm.max() + 1
        log("max time = %d"%(self.maxTime))
        num = multi_adj_time_norm.shape[0]
        multi_adj_time_norm = multi_adj_time_norm + sp.eye(num)
        print("uv graph link num = %d"%(multi_adj_time_norm.nnz))

        
        edge_src, edge_dst = multi_adj_time_norm.nonzero()
        time_seq = multi_adj_time_norm.tocoo().data
        self.time_seq_tensor = t.from_numpy(time_seq.astype(np.float64)).long().to(device_gpu)
        
        self.ratingClass = np.unique(trainMat.data).size
        log("user num =%d, item num =%d"%(self.userNum, self.itemNum))

        self.uv_g = dgl.graph(data=(edge_src, edge_dst),
                              idtype=t.int32,
                              num_nodes=multi_adj_time_norm.shape[0],
                              device=device_gpu)

        #train data
        train_u, train_v = self.trainMat.nonzero()
        assert np.sum(self.trainMat.data ==0) == 0
        log("train data size = %d"%(train_u.size))
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = BPRData(train_data, self.itemNum, self.trainMat, self.args.num_ng, True) # train data에 negative sampling 추가
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0)
        #valid data
        valid_dataset = BPRData(validData, self.itemNum, self.trainMat, 0, False)
        self.valid_loader  = dataloader.DataLoader(valid_dataset, batch_size=args.test_batch, shuffle=False, num_workers=0)
        
        self.lr = self.args.lr #0.001
        self.curEpoch = 0
        self.isLoadModel = isLoad
        #history
        self.train_loss = []
        self.his_hr = []
        self.his_rmse = []
        self.his_mae = []
        self.his_precision = []
        self.his_ndcg  = []
        gc.collect()
        log("gc.collect()")

    def setRandomSeed(self):
        np.random.seed(self.args.seed)
        t.manual_seed(self.args.seed)
        t.cuda.manual_seed(self.args.seed)
        random.seed(self.args.seed)
    
    def getData(self, args):
        if os.path.isfile(self.datasetDir + f'/uu_vv_graph_{args.seed}.pkl') and os.path.isfile(self.datasetDir + '/multi_item_adj.pkl') and (not args.regen_base) and (not args.regen_seed):
            with open(self.datasetDir + f'/uu_vv_graph_{args.seed}.pkl', 'rb') as fs:
                uu_vv_graph = pickle.load(fs)
            with open(self.datasetDir + '/multi_item_adj.pkl', 'rb') as fs:
                multi_adj_time = pickle.load(fs)
        else:
            print("Re-Creating Dataset")
            trainMat, test_data, validData, train_time, trust, uu_vv_graph, multi_adj_time = dp_main(
                args.dataset,
                args.seed,
                regen_base=args.regen_base,
                regen_seed=args.regen_seed,
            )
            
        data = loadData(args.dataset, seed=args.seed)
        trainMat, _, validData, _, _ = data
        uuMat = uu_vv_graph['UU'].astype(bool)
        iiMat = uu_vv_graph['II'].astype(bool)
        return trainMat, validData, multi_adj_time, uuMat, iiMat

    #初始化参数
    def prepareModel(self):
        self.modelName = self.getModelName() 
        self.setRandomSeed()

        self.layer = eval(self.args.layer)
        self.hide_dim = args.hide_dim
        self.out_dim = sum(self.layer) + self.hide_dim
        # self.out_dim = self.hide_dim
                
        # GCN model init
        self.model = MODEL(self.args, self.userNum, self.itemNum, self.hide_dim, \
            self.maxTime, self.ratingClass, self.layer).cuda()


        if self.args.dgi_graph_act == "sigmoid":
            dgiGraphAct = nn.Sigmoid()
        elif self.args.dgi_graph_act == "tanh":
            dgiGraphAct = nn.Tanh()

        self.uu_dgi = DGI(self.uu_graph, self.out_dim, self.out_dim, nn.PReLU(), dgiGraphAct).cuda()
        self.ii_dgi = DGI(self.ii_graph, self.out_dim, self.out_dim, nn.PReLU(), dgiGraphAct).cuda()
        
        self.opt = t.optim.Adam([
            {'params': self.model.parameters(), 'weight_decay': 0},
            {'params': self.uu_dgi.parameters(), 'weight_decay': 0},
            {'params': self.ii_dgi.parameters(), 'weight_decay': 0},
        ], lr=self.args.lr)

    def adjust_learning_rate(self, opt, epoch):
        for param_group in opt.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.minlr)
            # log("cur lr = %.6f"%(param_group['lr']))
    
    def innerProduct(self, u, i, j):
        pred_i = t.sum(t.mul(u,i), dim=1)
        pred_j = t.sum(t.mul(u,j), dim=1)
        return pred_i, pred_j
    
    def run(self):
        self.prepareModel()
        if self.isLoadModel == True:
            print('###############################test###############################')
            self.loadModel(LOAD_MODEL_PATH)
            RMSE, MAE, NDCG, Precision = self.test()
            print(f"TEST RMSE : {RMSE}, TEST MAE : {MAE}, TEST NDCG : {NDCG}, TEST Precision@{self.args.top_k} : {Precision}")
            return
        cvWait = 0
        best_epoch = 0
        best_NDCG = 0
        best_RMSE = 9999
        for e in range(self.curEpoch, self.args.epochs+1):
            self.curEpoch = e
            log("**************************************************************")
            log("start train")
            epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss = self.trainModel()
            log("end train")
            self.train_loss.append(epoch_loss)
            log("epoch %d/%d, epoch_loss=%.2f, dgi_uu_loss=%.4f, dgi_ii_loss=%.4f"% \
                (e, self.args.epochs, epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss))
            
            if e < self.args.startTest:
                RMSE, MAE, NDCG = 0, 0, 0
                cvWait = 0
            else:
                RMSE, MAE, NDCG, Precision = self.validModel(self.valid_loader)
                # self.his_hr.append(HR)
                self.his_rmse.append(RMSE)
                self.his_mae.append(MAE)
                self.his_ndcg.append(NDCG)
                self.his_precision.append(Precision)
                log("epoch %d/%d, valid RMSE = %.4f, valid MAE = %.4f, valid NDCG = %.4f, valid Precision@%d = %.4f"%(e, self.args.epochs, RMSE, MAE, NDCG, self.args.top_k, Precision))
            
            if e%10 == 0 and e != 0:
                testRMSE, testMAE, testNDCG, testPrecision = self.test()
                log("test RMSE = %.4f, test MAE = %.4f, test NDCG = %.4f, test Precision@%d = %.4f"%(testRMSE, testMAE, testNDCG, self.args.top_k, testPrecision))

            self.adjust_learning_rate(self.opt, e)
            if NDCG > best_NDCG:
                best_NDCG = NDCG
            # if RMSE < best_RMSE:
                # best_RMSE = RMSE
                cvWait = 0
                best_epoch = self.curEpoch
                self.saveModel()
            else:
                cvWait += 1
                log("cvWait = %d"%(cvWait))

            self.saveHistory()

            if cvWait == self.args.patience:
                log('Early stopping! best epoch = %d'%(best_epoch))
                self.loadModel(self.modelName)
                testRMSE, testMAE, testNDCG, testPrecision = self.test()
                log("test RMSE = %.4f, test MAE = %.4f, test NDCG = %.4f, test Precision@%d = %.4f"%(testRMSE, testMAE, testNDCG, self.args.top_k, testPrecision))
                break
        
        
    def test(self):
        #load test dataset
        with open(self.datasetDir + f"/test_data_{self.args.seed}.pkl", 'rb') as fs:
            test_data = pickle.load(fs)
        test_dataset = BPRData(test_data, self.itemNum, self.trainMat, 0, False)
        self.test_loader  = dataloader.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=0)
        RMSE, MAE, NDCG, Precision = self.validModel(self.test_loader)
        return RMSE, MAE, NDCG, Precision
    

    def trainModel(self):
        train_loader = self.train_loader
        log("start negative sample...")
        train_loader.dataset.ng_sample()
        log("finish negative sample...")
        epoch_loss = 0
        epoch_uu_dgi_loss = 0
        epoch_ii_dgi_loss = 0
        for user, item_i, item_j in train_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()
            user_embed, item_embed = self.model(self.uv_g, self.time_seq_tensor, self.out_dim, self.ratingClass, True)
            
            userEmbed = user_embed[user]
            posEmbed = item_embed[item_i]
            negEmbed = item_embed[item_j]

            pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)

            bprloss = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log().sum()
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = 0.5*(bprloss + self.args.reg * regLoss)/self.args.batch

            uu_dgi_loss = 0
            ii_dgi_loss = 0
            if self.args.lam[0] != 0:
                uu_dgi_pos_loss, uu_dgi_neg_loss = self.uu_dgi(user_embed, self.uu_subGraph_adj_tensor, \
                    self.uu_subGraph_adj_norm, self.uu_node_subGraph, self.uu_dgi_node)
                userMask = t.zeros(self.userNum).cuda()
                userMask[user] = 1
                userMask = userMask * self.uu_dgi_node_mask
                uu_dgi_loss = ((uu_dgi_pos_loss*userMask).sum() + (uu_dgi_neg_loss*userMask).sum())/t.sum(userMask)
                epoch_uu_dgi_loss += uu_dgi_loss.item()

            if self.args.lam[1] != 0:
                ii_dgi_pos_loss, ii_dgi_neg_loss = self.ii_dgi(item_embed, self.ii_subGraph_adj_tensor, \
                    self.ii_subGraph_adj_norm, self.ii_node_subGraph, self.ii_dgi_node)
                iiMask = t.zeros(self.itemNum).cuda()
                iiMask[item_i] = 1
                iiMask[item_j] = 1
                iiMask = iiMask * self.ii_dgi_node_mask
                ii_dgi_loss = ((ii_dgi_pos_loss*iiMask).sum() + (ii_dgi_neg_loss*iiMask).sum())/t.sum(iiMask)
                epoch_ii_dgi_loss += ii_dgi_loss.item()
            loss = loss + self.args.lam[0] * uu_dgi_loss + self.args.lam[1] * ii_dgi_loss

            epoch_loss += bprloss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # log('setp %d/%d, step_loss = %f'%(i, loss.item()), save=False, oneline=True)
        return epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss

    def validModel(self, data_loader, save=False):
        self.model.eval()
        with t.no_grad():
            HR, NDCG = [], []
            u_lst, i_lst, p_lst, r_lst = [],[],[],[]
            user_embed, item_embed = self.model(self.uv_g, self.time_seq_tensor, self.out_dim, self.ratingClass, True)
            for user, item_i, rating in data_loader:
                user = user.long().cuda()
                item_i = item_i.long().cuda()
                rating = rating.float().cuda()
                userEmbed = user_embed[user]
                testItemEmbed = item_embed[item_i]
                pred_i = t.sum(t.mul(userEmbed, testItemEmbed), dim=1)
                # user, item, pred, rating -> 모두 모아서 dataframe으로 만들고, ndcg+precision계산
                u_lst.append(user); i_lst.append(item_i); r_lst.append(rating); p_lst.append(pred_i)
            
        total_user = t.cat(u_lst, axis=0)
        total_item = t.cat(i_lst, axis=0)
        total_rating = t.cat(r_lst, axis=0)
        total_pred = t.cat(p_lst, axis=0)
        del u_lst, i_lst, r_lst, p_lst
        t.cuda.empty_cache()
        # pred값 -> 0~5의 range로 변환
        total_pred = (total_pred-total_pred.min())/(total_pred.max()-total_pred.min()+1e-9) * 5
        # RMSE/MAE 계산
        r_mask = (total_rating!=0)
        RMSE = t.sqrt(F.mse_loss(total_pred[r_mask], total_rating[r_mask]))
        MAE = F.l1_loss(total_pred[r_mask], total_rating[r_mask])
        # NDCG/Precision 계산 (per-user top-k with discount on rank)
        unique_users = total_user.unique()
        discount = t.reciprocal(t.log2(t.arange(self.args.top_k, device=total_pred.device, dtype=t.float32)+2))
        dcg_total = t.tensor(0.0, device=total_pred.device)
        idcg_total = t.tensor(0.0, device=total_pred.device)
        precision_total = 0.0
        for user in unique_users:
            u_idx = (total_user == user).nonzero(as_tuple=True)[0]
            if u_idx.numel() == 0:
                continue
            user_pred = total_pred[u_idx]
            user_rating = total_rating[u_idx]
            # DCG
            topk = min(self.args.top_k, user_pred.numel())
            _, dcg_indices = user_pred.topk(topk)
            dcg_total += (user_rating[dcg_indices] * discount[:topk]).sum()
            hits = (user_rating[dcg_indices] > 0).float().sum()
            precision_total += (hits / topk).item()
            # IDCG
            ideal_topk = min(self.args.top_k, user_rating.numel())
            sorted_rating, _ = user_rating.topk(ideal_topk)
            idcg_total += (sorted_rating * discount[:ideal_topk]).sum()
        NDCG = (dcg_total / (idcg_total + 1e-8)).item()
        Precision = precision_total / len(unique_users) if len(unique_users) > 0 else 0.0
        return RMSE.item(), MAE.item(), NDCG, Precision

    def getModelName(self):
        title = "KCGN_"
        # ModelName = title + self.args.dataset + "_" + modelUTCStr + \
        ModelName = title + self.args.dataset + "_" + \
        "_reg_" + str(self.args.reg)+ \
        "_batch_" + str(self.args.batch) + \
        "_lr_" + str(self.args.lr) + \
        "_decay_" + str(self.args.decay) + \
        "_hide_" + str(self.args.hide_dim) + \
        "_Layer_" + self.args.layer +\
        "_slope_" + str(self.args.slope) +\
        "_top_" + str(self.args.top_k) +\
        "_fuse_" + self.args.fuse +\
        "_timeStep_" + str(self.args.time_step) +\
        "_lam_" + str(self.args.lam) + str(self.args.dgi_graph_act)
        return ModelName


    def saveHistory(self):
        #保存历史数据，用于画图
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['RMSE'] = self.his_rmse
        history['MAE'] = self.his_mae
        history['NDCG'] = self.his_ndcg
        history['Precision'] = self.his_precision
        ModelName = self.modelName

        with open(r'./History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self):
        # ModelName = self.getModelName()
        ModelName = self.modelName
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['RMSE'] = self.his_rmse
        history['MAE'] = self.his_mae
        history['NDCG'] = self.his_ndcg
        history['Precision'] = self.his_precision
        savePath = r'./Model/' + self.args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            'lr': self.lr,
            'model': self.model,
            'reg':self.args.reg,
            'history':history,
            }
        t.save(params, savePath)


    def loadModel(self, modelPath):
        checkpoint = t.load(r'./Model/' + args.dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.lr = checkpoint['lr']
        self.model = checkpoint['model']
        self.args.reg = checkpoint['reg']
        #恢复history
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history.get('HR', [])
        self.his_rmse = history.get('RMSE', [])
        self.his_mae = history.get('MAE', [])
        self.his_ndcg = history.get('NDCG', [])
        self.his_precision = history.get('Precision', [])
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KCGN main.py')
    #dataset params
    parser.add_argument('--regen', action='store_true')
    parser.add_argument('--regen_base', action='store_true')
    parser.add_argument('--regen_seed', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--dataset', type=str, default="Yelp", help="Epinions,Yelp")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--hide_dim', type=int, default=64)
    parser.add_argument('--layer', type=str, default="[64]")
    parser.add_argument('--slope', type=float, default=0.4)

    parser.add_argument('--reg', type=float, default=0.05)
    parser.add_argument('--decay', type=float, default=0.98)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--minlr', type=float, default=0.0001)
    parser.add_argument('--test_batch', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=180)
    #early stop params
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_ng', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--fuse', type=str, default="mean", help="mean or weight")

    parser.add_argument('--dgi_graph_act', type=str, default="sigmoid", help="sigmoid or tanh")
    parser.add_argument('--lam', type=str, default='[0.1,0.001]')
    parser.add_argument('--subNode', type=int, default=10)

    parser.add_argument('--time_step', type=float, default=360)
    parser.add_argument('--startTest', type=int, default=0)
    
    parser.add_argument('--id', type=int, default=0)

    args = parser.parse_args()
    args.lam = eval(args.lam)
    assert len(args.lam) == 2
    args.regen_base = args.regen_base or args.regen
    args.regen_seed = args.regen_seed or args.regen
    print(args)
    mkdir(args.dataset)
    device_gpu = t.device(f"cuda:{args.id}")
    print(device_gpu)
    if args.regen_base:
        process_kcgn(args.dataset, args.seed)
    if args.eval:
        isLoadModel=True
    hope = Model(args, isLoadModel)
    modelName = hope.getModelName()
    model_dir = os.path.join('Model',args.dataset)
    model_list = sorted(os.listdir(model_dir))
    LOAD_MODEL_PATH = ''.join(model_list[-1].split('.pth'))
    print(LOAD_MODEL_PATH)
    
    print('ModelNmae = ' + modelName)
    
    hope.run()
    hope.test()
