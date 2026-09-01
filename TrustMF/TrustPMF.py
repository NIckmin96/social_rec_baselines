import torch
import torch.nn as nn

class TrustPMF(nn.Module):
    def __init__(self, num_users, num_items, d_model):
        super().__init__()
        self.num_users = num_users
        self.beta1, self.beta2 = 0.5, 0.5
        self.U = nn.Embedding(num_users+2, d_model, padding_idx=num_users+1)
        self.V = nn.Embedding(num_items+1, d_model)
        self.B = nn.Embedding(num_users+2, d_model, padding_idx=num_users+1)
        self.W = nn.Embedding(num_users+2, d_model, padding_idx=num_users+1)

    def forward(self, x):
        # rating pred
        r_hat = torch.sigmoid(torch.matmul(self.U(x['user_id']), self.V(x['product_id']).transpose(0,1)))

        # trust pred
        trustee_emb = self.W(x['trustee'])
        t_hat_trustee = self.B(x['user_id']).unsqueeze(1).expand(*trustee_emb.size())
        t_hat_trustee = torch.sigmoid(torch.matmul(t_hat_trustee, trustee_emb.transpose(1,2)))

        truster_emb = self.W(x['truster'])
        t_hat_truster = self.B(x['user_id']).unsqueeze(1).expand(*truster_emb.size())
        t_hat_truster = torch.sigmoid(torch.matmul(t_hat_truster, truster_emb.transpose(1,2)))

        # reg term
        reg_term = (self.beta1/2)*torch.sum(torch.pow(self.U.weight-self.B.weight, 2))
        reg_term += (self.beta2/2)*torch.sum(torch.pow(self.U.weight-self.W.weight, 2))
        reg_term += torch.sum(torch.pow(self.U.weight, 2))/2 # U frobenius
        reg_term += torch.sum(torch.pow(self.V.weight, 2))/2 # V frobenius
        reg_term += torch.sum(torch.pow(self.B.weight, 2))/2 # B frobenius
        reg_term += torch.sum(torch.pow(self.W.weight, 2))/2 # W frobenius

        return torch.diagonal(r_hat), torch.diagonal(t_hat_truster, dim1=-2, dim2=-1), torch.diagonal(t_hat_trustee, dim1=-2, dim2=-1), reg_term
    