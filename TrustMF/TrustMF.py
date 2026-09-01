import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrustMF(nn.Module):
    def __init__(self, truster, trustee):
        super().__init__()
        if isinstance(truster, nn.DataParallel):
            self.truster = truster.module
        else:
            self.truster = truster
            
        if isinstance(trustee, nn.DataParallel):
            self.trustee = trustee.module
        else:
            self.trustee = trustee
        
    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        # Eq. (6): g( ((B_i^r)^T V_j^r + (W_i^e)^T V_j^e) / 2 ) * R_max
        # per-pair dot products only -- no B x B matmul + diag.
        logit = torch.sum(self.truster.B(x['user_id']) * self.truster.V(x['product_id']), dim=-1)
        logit = logit + torch.sum(self.trustee.W(x['user_id']) * self.trustee.V(x['product_id']), dim=-1)
        rating_pred = torch.sigmoid(logit / 2) * 5
        # A-5: Algorithm 1 clips the final prediction to [1, R_max].
        rating_pred = rating_pred.clamp(1.0, 5.0)

        return rating_pred
        
class Truster(nn.Module):
    def __init__(self, num_users, num_items, d_model, n_b, n_v, m_b, m_w):
        super().__init__()
        self.num_users = num_users
        self.B = nn.Embedding(num_users+2, d_model, padding_idx=num_users+1)
        self.W = nn.Embedding(num_users+2, d_model, padding_idx=num_users+1)
        self.V = nn.Embedding(num_items+1, d_model)
        self.B.weight.data.normal_(mean=0.0, std=0.1)
        self.W.weight.data.normal_(mean=0.0, std=0.1)
        self.V.weight.data.normal_(mean=0.0, std=0.1)
        # self.n_b, self.n_v = n_b, n_v
        # self.m_b, self.m_w = m_b, m_w
        self.register_buffer("n_b", n_b)
        self.register_buffer("n_v", n_v)
        self.register_buffer("m_b", m_b)
        self.register_buffer("m_w", m_w)
        
    def forward(self, user_id, product_id):
        # rating term of Eq. (4): g(B_i^T V_j), one score per (user, item) pair.
        return torch.sigmoid(torch.sum(self.B(user_id) * self.V(product_id), dim=-1))

    def trust_forward(self, i, k):
        # trust term of Eq. (4): g(B_i^T W_k) for observed edges (i, k) in Psi.
        return torch.sigmoid(torch.sum(self.B(i) * self.W(k), dim=-1))

    def reg_loss(self):
        # A-2: weighted-lambda regularisation of Eq. (4), summed over every user i,
        #      item j and trusted neighbour k exactly once -- called once per epoch,
        #      not once per rating row.
        device = self.B.weight.device
        u = torch.arange(self.num_users, device=device)
        v = torch.arange(self.V.num_embeddings - 1, device=device)
        reg_B = torch.sum((self.n_b[u] + self.m_b[u]).float() * torch.sum(self.B.weight[u] ** 2, dim=-1))
        reg_V = torch.sum(self.n_v[v].float() * torch.sum(self.V.weight[v] ** 2, dim=-1))
        reg_W = torch.sum(self.m_w[u].float() * torch.sum(self.W.weight[u] ** 2, dim=-1))
        return reg_B + reg_V + reg_W

    
class Trustee(nn.Module):
    def __init__(self, num_users, num_items, d_model, n_b, n_v, m_b, m_w):
        super().__init__()
        self.num_users = num_users
        self.B = nn.Embedding(num_users+2, d_model, padding_idx=num_users+1)
        self.W = nn.Embedding(num_users+2, d_model, padding_idx=num_users+1)
        self.V = nn.Embedding(num_items+1, d_model)
        self.B.weight.data.normal_(mean=0.0, std=0.1)
        self.W.weight.data.normal_(mean=0.0, std=0.1)
        self.V.weight.data.normal_(mean=0.0, std=0.1)
        # self.n_w, self.n_v = n_b, n_v
        # self.m_b, self.m_w = m_b, m_w
        self.register_buffer("n_w", n_b)
        self.register_buffer("n_v", n_v)
        self.register_buffer("m_b", m_b)
        self.register_buffer("m_w", m_w)
        
    def forward(self, user_id, product_id):
        # rating term of Eq. (5): g(W_i^T V_j), one score per (user, item) pair.
        return torch.sigmoid(torch.sum(self.W(user_id) * self.V(product_id), dim=-1))

    def trust_forward(self, i, k):
        # trust term of Eq. (5): g(B_k^T W_i) for observed edges (k, i) in Psi,
        # i.e. k trusts i.
        return torch.sigmoid(torch.sum(self.B(k) * self.W(i), dim=-1))

    def reg_loss(self):
        # A-2: weighted-lambda regularisation of Eq. (5), summed over every user i,
        #      item j and truster k exactly once -- called once per epoch.
        device = self.W.weight.device
        u = torch.arange(self.num_users, device=device)
        v = torch.arange(self.V.num_embeddings - 1, device=device)
        reg_W = torch.sum((self.n_w[u] + self.m_w[u]).float() * torch.sum(self.W.weight[u] ** 2, dim=-1))
        reg_V = torch.sum(self.n_v[v].float() * torch.sum(self.V.weight[v] ** 2, dim=-1))
        reg_B = torch.sum(self.m_b[u].float() * torch.sum(self.B.weight[u] ** 2, dim=-1))
        return reg_W + reg_V + reg_B
