import torch
import torch.nn as nn

class SocialMF(nn.Module):
    """
    - user latent feature(embedding) : U
    - item latent feature(embedding) : V
    - Regularization coefficients : [lambda_u, lambda_t, lambda_v]
    - neighbor_indices [bs, t]
    - neighbor_values [bs, t]

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_users, num_items, d_model, lambda_u=0.1, lambda_v=0.1, lambda_t=5.0):
        super().__init__()
        self.U = nn.Embedding(num_users+1, d_model, padding_idx=0)
        self.V = nn.Embedding(num_items+1, d_model, padding_idx=0)
        # paper: "initial values of U and V are samples from normal noises with zero mean".
        # small std keeps U^T V near 0 at start so g() is not saturated (g' ~ 0) -- matches
        # the TrustMF sibling impl and avoids vanishing reconstruction gradients.
        self.U.weight.data.normal_(mean=0.0, std=0.1)
        self.V.weight.data.normal_(mean=0.0, std=0.1)
        self.U.weight.data[0].zero_()  # restore padding_idx row after manual init
        self.V.weight.data[0].zero_()
        self.lambda_u, self.lambda_v, self.lambda_t = lambda_u, lambda_v, lambda_t
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_idx, item_idx):
        # rating prediction: g(U_u^T V_i). SocialMF does not need neighbor info at
        # prediction time (unlike STE), so this only touches U and V.
        u = self.U(user_idx)
        v = self.V(item_idx)
        r_hat = self.sigmoid((u * v).sum(dim=1)).view(-1, 1)
        return r_hat

    def item_reg_loss(self, item_idx):
        v = self.V(item_idx)
        return self.lambda_v * torch.pow(v, 2).sum(dim=1)

    def user_reg_loss(self, user_idx, neighbor_indices, neighbor_values):
        # zero-mean prior on U_u plus the trust-propagation term that pulls U_u toward
        # the trust-weighted average of its direct neighbors' feature vectors (eq. 6/9).
        u = self.U(user_idx)
        reg_u = self.lambda_u * torch.pow(u, 2).sum(dim=1)

        neighbors = self.U(neighbor_indices)  # [bs, t, d]
        neighbors = neighbors.permute(0, 2, 1)  # [bs, d, t]
        neighbor_values = neighbor_values.unsqueeze(-1)  # [bs, t, 1]
        u_hat = torch.matmul(neighbors, neighbor_values).squeeze(-1)  # [bs, d]
        reg_t = self.lambda_t * torch.pow(u - u_hat, 2).sum(dim=1)

        return reg_u, reg_t
