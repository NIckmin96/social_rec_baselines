import sys
import torch
# import pandas as pd
# import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TrustSVD(nn.Module):
    def __init__(self, num_users, num_items, d_model, mu, lambda_, lambda_t, device):
        super().__init__()
        self.lambda_ = lambda_
        self.lambda_t = lambda_t
        self.device = device
        self.mu = mu
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        self.user_embedding = nn.Embedding(num_users, d_model)
        self.item_embedding = nn.Embedding(num_items, d_model)
        
        self.user_feature = nn.Embedding(num_users, d_model)
        self.item_feature = nn.Embedding(num_items, d_model)
                
    def forward(self, x):
        user_id = x['user_id']
        product_id = x['product_id']
        i_u = x['i_u']
        t_u = x['t_u']

        len_t_v = x['len_t_v']
        len_u_j = x['len_u_j']
        len_u_i = x['len_u_i']

        p_u = self.user_embedding(user_id) # bs x d
        q_j = self.item_embedding(product_id) # bs x d
        b_u = self.user_bias(user_id).view(-1,1)
        b_j = self.item_bias(product_id).view(-1,1)
        # r hat
        r_hat = b_u+b_j+self.mu

        y_i = self.item_feature(i_u) # bs x n x d
        i_mask = (i_u != 0).unsqueeze(-1) # bs x n x 1
        len_iu = i_mask.sum(dim=1).clamp(min=1).sqrt()
        inv_len_iu = torch.reciprocal(len_iu)
        sum_y_i = (y_i * i_mask).sum(dim=1) * inv_len_iu # bs x d

        w_v = self.user_feature(t_u) # bs x u x d
        t_mask = (t_u != 0).unsqueeze(-1) # bs x u x 1
        len_tu = t_mask.sum(dim=1).clamp(min=1).sqrt()
        inv_len_tu = torch.reciprocal(len_tu)
        sum_w_v = (w_v * t_mask).sum(dim=1) * inv_len_tu # bs x d

        # compute per-sample inner product without forming a full BxB matrix
        # print(r_hat.size(), (p_u + sum_y_i + sum_w_v).size(), q_j.size())
        r_hat = r_hat + torch.sum((p_u + sum_y_i + sum_w_v) * q_j, dim=1, keepdim=True) # bs

        # t hat
        t_hat = torch.matmul(p_u.unsqueeze(1), w_v.transpose(2,1)).squeeze(1)

        # reg term
        reg_term = 0.5*self.lambda_*torch.sum(inv_len_iu.view(-1,1)*torch.pow(b_u, 2))

        len_uj = torch.pow(len_u_j, 0.5)
        inv_len_uj = torch.where(len_uj==0, torch.zeros_like(len_uj), torch.reciprocal(len_uj))
        reg_term += 0.5*self.lambda_*torch.sum(inv_len_uj.view(-1,1)*torch.pow(b_j, 2))

        reg_term += torch.sum((0.5*self.lambda_*inv_len_iu.view(-1,1) + 0.5*self.lambda_*inv_len_tu.view(-1,1))*torch.pow(p_u, 2))
        reg_term += 0.5*self.lambda_*torch.sum(inv_len_uj.view(-1,1)*torch.pow(q_j, 2))

        len_ui = torch.pow(len_u_i, 0.5)
        inv_len_ui = torch.where(len_ui==0, torch.zeros_like(len_ui), torch.reciprocal(len_ui))
        reg_term += 0.5*self.lambda_*torch.sum(inv_len_ui.view(*inv_len_ui.size(),1)*torch.pow(y_i, 2))

        len_tv = torch.pow(len_t_v, 0.5)
        inv_len_tv = torch.where(len_tv==0, torch.zeros_like(len_tv), torch.reciprocal(len_tv))
        reg_term += 0.5*self.lambda_*torch.sum(inv_len_tv.view(*inv_len_tv.size(),1)*torch.pow(w_v, 2))

        return r_hat, t_hat, reg_term
        
