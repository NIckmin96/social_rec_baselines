# coding=UTF-8
"""
Explicit-feedback variant of KCGN.

The original main.py (paper eq. 8) trains purely on a BPR pairwise ranking
loss: positive = any observed (user, item) pair, negative = an unobserved
item, with no notion of *which* rating the positive got. The rating value
(1~5) only ever enters the model via graph construction (create_adj.py builds
one item sub-vertex per rating class), so the learned score is never
supervised to reproduce a rating magnitude, and the RMSE/MAE that the
original validModel() reports is a post-hoc global min-max rescaling of that
ranking score -- not a calibrated rating prediction.

This script keeps the same GNN encoder (MyGCN.MODEL) and DGI mutual-info
regularizers, but replaces the BPR term with a direct rating-regression term:
pred = sigmoid(u . v), trained via MSE against rating/Rmax (same
parameterization TrustMF/SocialMF use), so RMSE/MAE are the metric the model
was actually optimized for.
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ToolScripts.TimeLogger import log
import pickle
import os
import gc
import random
import argparse
import scipy.sparse as sp
from ToolScripts.utils import loadData
from ToolScripts.utils import load
from ToolScripts.utils import buildSubGraph
from ToolScripts.utils import sparse_mx_to_torch_sparse_tensor
from ToolScripts.utils import mkdir
import dgl
from MyGCN import MODEL
from BPRData import BPRData
import torch.utils.data as dataloader
from DGI.dgi import DGI

from dataProcess import main as dp_main
from data_preprocess import process_kcgn

import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common_preprocess.metrics import rmse as _rmse, mae as _mae, dump_predictions  # noqa: E402


device_gpu = t.device("cuda")

isLoadModel = False
LOAD_MODEL_PATH = ""

class Model():

    def __init__(self, args, isLoad=False):
        self.args = args
        self.datasetDir = os.path.join(os.getcwd(), "dataset", args.dataset)
        trainMat, validData, multi_adj_time, uuMat, iiMat = self.getData(args)
        self.userNum, self.itemNum = trainMat.shape
        self.Rmax = float(trainMat.data.max())
        log("uu num = %d"%(uuMat.nnz))
        log("ii num = %d"%(iiMat.nnz))
        self.trainMat = trainMat

        uuMat_edge_src, uuMat_edge_dst = uuMat.nonzero()
        self.uu_graph = dgl.graph(data=(uuMat_edge_src, uuMat_edge_dst),
                            idtype=t.int32,
                            num_nodes=uuMat.shape[0],
                            device=device_gpu)
        iiMat_edge_src, iiMat_edge_dst = iiMat.nonzero()
        self.ii_graph = dgl.graph(data=(iiMat_edge_src, iiMat_edge_dst),
                            idtype=t.int32,
                            num_nodes=iiMat.shape[0],
                            device=device_gpu)

        # get sub graph message (unchanged from main.py: DGI needs connected
        # sub-graph pooling regardless of the downstream task)
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

        # norm time value (unchanged from main.py)
        log("time process")
        self.time_step = self.args.time_step
        log("time step = %.1f hour"%(self.time_step))
        time_step = 3600*self.time_step
        row, col = multi_adj_time.nonzero()
        data = multi_adj_time.data
        minUTC = data.min()
        data = ((data-minUTC)/time_step).astype(np.int64)+2
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
        log("user num =%d, item num =%d, Rmax=%.1f"%(self.userNum, self.itemNum, self.Rmax))

        self.uv_g = dgl.graph(data=(edge_src, edge_dst),
                              idtype=t.int32,
                              num_nodes=multi_adj_time_norm.shape[0],
                              device=device_gpu)

        # rating-regression training data: (user, item, rating) triples, no negative
        # sampling. Unlike BPR, the model here is directly supervised on rating
        # magnitude, so it has no use for unobserved-item negatives at train time.
        train_coo = self.trainMat.tocoo()
        train_u, train_v, train_r = train_coo.row, train_coo.col, train_coo.data
        assert np.sum(train_r == 0) == 0
        log("train data size = %d"%(train_u.size))
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1), train_r.reshape(-1,1))).tolist()
        train_dataset = BPRData(train_data, self.itemNum, self.trainMat, 0, False)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0)
        # valid data: BPRData already yields (user, item, rating) triples in
        # is_training=False mode, and validData carries the 100 negative-sampled
        # rating==0 rows from dataProcess.testNegSample -- validModel() filters those
        # out before computing RMSE/MAE.
        valid_dataset = BPRData(validData, self.itemNum, self.trainMat, 0, False)
        self.valid_loader  = dataloader.DataLoader(valid_dataset, batch_size=args.test_batch, shuffle=False, num_workers=0)

        self.lr = self.args.lr
        self.curEpoch = 0
        self.isLoadModel = isLoad
        self.train_loss = []
        self.his_rmse = []
        self.his_mae = []
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

    def prepareModel(self):
        self.modelName = self.getModelName()
        self.setRandomSeed()

        self.layer = eval(self.args.layer)
        self.hide_dim = self.args.hide_dim
        self.out_dim = sum(self.layer) + self.hide_dim

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

    def run(self):
        self.prepareModel()
        if self.isLoadModel == True:
            print('###############################test###############################')
            self.loadModel(LOAD_MODEL_PATH)
            RMSE, MAE = self.test()
            print(f"TEST RMSE : {RMSE}, TEST MAE : {MAE}")
            return
        cvWait = 0
        best_epoch = 0
        best_RMSE = 9999
        for e in range(self.curEpoch, self.args.epochs+1):
            self.curEpoch = e
            log("**************************************************************")
            log("start train")
            epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss = self.trainModel()
            log("end train")
            self.train_loss.append(epoch_loss)
            log("epoch %d/%d, epoch_loss=%.4f, dgi_uu_loss=%.4f, dgi_ii_loss=%.4f"% \
                (e, self.args.epochs, epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss))

            if e < self.args.startTest:
                RMSE, MAE = 9999, 9999
                cvWait = 0
            else:
                RMSE, MAE = self.validModel(self.valid_loader)
                self.his_rmse.append(RMSE)
                self.his_mae.append(MAE)
                log("epoch %d/%d, valid RMSE = %.4f, valid MAE = %.4f"%(e, self.args.epochs, RMSE, MAE))

            if e%10 == 0 and e != 0:
                testRMSE, testMAE = self.test()
                log("test RMSE = %.4f, test MAE = %.4f"%(testRMSE, testMAE))

            self.adjust_learning_rate(self.opt, e)
            if RMSE < best_RMSE:
                best_RMSE = RMSE
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
                testRMSE, testMAE = self.test()
                log("test RMSE = %.4f, test MAE = %.4f"%(testRMSE, testMAE))
                break

    def test(self):
        with open(self.datasetDir + f"/test_data_{self.args.seed}.pkl", 'rb') as fs:
            test_data = pickle.load(fs)
        test_dataset = BPRData(test_data, self.itemNum, self.trainMat, 0, False)
        self.test_loader  = dataloader.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=0)
        RMSE, MAE = self.validModel(self.test_loader, dump=(self.args.dataset, self.args.seed))
        return RMSE, MAE

    def trainModel(self):
        train_loader = self.train_loader
        epoch_loss = 0
        epoch_uu_dgi_loss = 0
        epoch_ii_dgi_loss = 0
        for user, item, rating in train_loader:
            user = user.long().cuda()
            item = item.long().cuda()
            rating = rating.float().cuda()
            user_embed, item_embed = self.model(self.uv_g, self.time_seq_tensor, self.out_dim, self.ratingClass, True)

            userEmbed = user_embed[user]
            itemEmbed = item_embed[item]

            # rating-regression term (replaces the BPR ranking term in main.py): predict
            # the normalized rating via a sigmoid-bounded inner product and fit it with
            # MSE, the same parameterization TrustMF/SocialMF use.
            pred = t.sigmoid(t.sum(userEmbed * itemEmbed, dim=1))
            target = rating / self.Rmax
            ratingLoss = F.mse_loss(pred, target, reduction='sum')
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(itemEmbed) ** 2)

            loss = 0.5*(ratingLoss + self.args.reg * regLoss)/self.args.batch

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
                iiMask[item] = 1
                iiMask = iiMask * self.ii_dgi_node_mask
                ii_dgi_loss = ((ii_dgi_pos_loss*iiMask).sum() + (ii_dgi_neg_loss*iiMask).sum())/t.sum(iiMask)
                epoch_ii_dgi_loss += ii_dgi_loss.item()
            loss = loss + self.args.lam[0] * uu_dgi_loss + self.args.lam[1] * ii_dgi_loss

            epoch_loss += ratingLoss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss

    def validModel(self, data_loader, dump=None):
        self.model.eval()
        with t.no_grad():
            p_lst, r_lst, u_lst, i_lst = [], [], [], []
            user_embed, item_embed = self.model(self.uv_g, self.time_seq_tensor, self.out_dim, self.ratingClass, True)
            for user, item, rating in data_loader:
                user = user.long().cuda()
                item = item.long().cuda()
                rating = rating.float().cuda()
                userEmbed = user_embed[user]
                itemEmbed = item_embed[item]
                # sigmoid bounds the score to [0,1], and it was trained (via MSE above)
                # to match rating/Rmax directly -- so *Rmax here is the calibrated
                # inverse of that mapping, not a post-hoc rescale over the eval batch.
                pred = t.sigmoid(t.sum(userEmbed * itemEmbed, dim=1)) * self.Rmax
                r_lst.append(rating)
                p_lst.append(pred)
                u_lst.append(user)
                i_lst.append(item)
        self.model.train()
        total_rating = t.cat(r_lst, axis=0)
        total_pred = t.cat(p_lst, axis=0)
        total_u = t.cat(u_lst, axis=0)
        total_i = t.cat(i_lst, axis=0)
        del r_lst, p_lst, u_lst, i_lst
        t.cuda.empty_cache()
        # exclude the injected negative-sampling placeholders (rating==0) that
        # dataProcess.testNegSample adds for ranking evaluation -- irrelevant here
        r_mask = (total_rating != 0)
        # KCGN ids are the common ids shifted to 0-based -> +1 to report in canonical space
        if dump is not None:
            ds, seed = dump
            dump_predictions("kcgn", ds, seed,
                             (total_u[r_mask] + 1), (total_i[r_mask] + 1),
                             total_rating[r_mask], total_pred[r_mask])
        RMSE = _rmse(total_rating[r_mask], total_pred[r_mask])
        MAE = _mae(total_rating[r_mask], total_pred[r_mask])
        return RMSE, MAE

    def getModelName(self):
        title = "KCGN_explicit_"
        ModelName = title + self.args.dataset + "_" + \
        "_reg_" + str(self.args.reg)+ \
        "_batch_" + str(self.args.batch) + \
        "_lr_" + str(self.args.lr) + \
        "_decay_" + str(self.args.decay) + \
        "_hide_" + str(self.args.hide_dim) + \
        "_Layer_" + self.args.layer +\
        "_slope_" + str(self.args.slope) +\
        "_fuse_" + self.args.fuse +\
        "_timeStep_" + str(self.args.time_step) +\
        "_lam_" + str(self.args.lam) + str(self.args.dgi_graph_act)
        return ModelName

    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        history['RMSE'] = self.his_rmse
        history['MAE'] = self.his_mae
        ModelName = self.modelName

        with open(r'./History/' + self.args.dataset + r'_explicit/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self):
        ModelName = self.modelName
        history = dict()
        history['loss'] = self.train_loss
        history['RMSE'] = self.his_rmse
        history['MAE'] = self.his_mae
        savePath = r'./Model/' + self.args.dataset + r'_explicit/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            'lr': self.lr,
            'model': self.model,
            'reg':self.args.reg,
            'history':history,
            }
        t.save(params, savePath)

    def loadModel(self, modelPath):
        checkpoint = t.load(r'./Model/' + self.args.dataset + r'_explicit/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.lr = checkpoint['lr']
        self.model = checkpoint['model']
        self.args.reg = checkpoint['reg']
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_rmse = history.get('RMSE', [])
        self.his_mae = history.get('MAE', [])
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KCGN explicit-feedback main.py')
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
    parser.add_argument('--patience', type=int, default=5)
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
    mkdir(args.dataset + "_explicit")
    device_gpu = t.device(f"cuda:{args.id}")
    print(device_gpu)
    if args.regen_base:
        process_kcgn(args.dataset, args.seed)
    isLoadModel = args.eval
    if isLoadModel:
        model_dir = os.path.join('Model', args.dataset + "_explicit")
        model_list = sorted(os.listdir(model_dir))
        LOAD_MODEL_PATH = ''.join(model_list[-1].split('.pth'))
        print(LOAD_MODEL_PATH)

    hope = Model(args, isLoadModel)
    print('ModelName = ' + hope.getModelName())

    hope.run()
    hope.test()
