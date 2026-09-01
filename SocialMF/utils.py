import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def minmax_scale_ratings(ratings, min_rating=1.0, max_rating=5.0):
    neg_mask = (ratings==0)
    denom = max_rating - min_rating
    if denom <= 0:
        raise ValueError("max_rating must be greater than min_rating")

    res = (ratings-min_rating) / denom
    res[neg_mask] = 0

    return res

class MyDataset(Dataset):
    """Rating-level dataset for the reconstruction term (eq. 1/12): one sample per
    observed (user, item) rating. Prediction does not need neighbor info."""
    def __init__(self, rating_df):
        self.user_id = torch.from_numpy(rating_df['user_id'].values).long()
        self.item_id = torch.from_numpy(rating_df['product_id'].values).long()
        raw = torch.from_numpy(rating_df['rating'].values).float()
        self.rating_raw = raw
        self.rating = minmax_scale_ratings(raw)

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        return self.user_id[idx], self.item_id[idx], self.rating[idx], self.rating_raw[idx]

class UserGraphDataset(Dataset):
    """User-level dataset for the regularization terms (eq. 12): one sample per user
    (rated or not), so every user's own feature vector is pulled toward its neighbors'
    weighted average, matching the paper's sum over u=1..N rather than over ratings."""
    def __init__(self, num_users, trust_mat):
        self.user_id = torch.arange(1, num_users + 1, dtype=torch.long)
        self.trust_mat = trust_mat

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        user_idx = self.user_id[idx]
        start = self.trust_mat.indptr[user_idx]
        end = self.trust_mat.indptr[user_idx+1]
        # trust weights are normalized floats in [0,1]; must stay float, not be truncated to int
        neighbor_values = torch.FloatTensor(self.trust_mat.data[start:end])
        neighbor_indices = torch.LongTensor(self.trust_mat.indices[start:end])

        return user_idx, neighbor_indices, neighbor_values

class ItemDataset(Dataset):
    """Item-level dataset for the regularization term (eq. 12): one sample per item,
    so ||V_i||^2 is summed once per unique item, matching the paper's sum over
    i=1..M rather than once per rating (which would over-count popular items)."""
    def __init__(self, num_items):
        self.item_id = torch.arange(1, num_items + 1, dtype=torch.long)

    def __len__(self):
        return len(self.item_id)

    def __getitem__(self, idx):
        return self.item_id[idx]

class Metrics:
    def __init__(self):
        pass

    def RMSE(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))

    def apply_softmax(self, group):
        logits = torch.tensor(group['pred'].values, dtype=torch.float)
        logits = F.softmax(logits, dim=0)
        group['logits'] = logits
        return group

    def get_logits(self, df):
        df = df.groupby('user_id').apply(self.apply_softmax).reset_index(drop=True)
        return df

    def rank_metrics(self, items, logits, ratings, k=10):
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
        item_mask = torch.tensor(list(map(lambda x:1 if x.item() in set(gt_items.tolist()) else 0, list(recommended_i))))

        # DCG = torch.sum(recommended_r*item_mask/discount)
        DCG = torch.sum(recommended_r/discount)
        IDCG = torch.sum(ideal_r/discount)
        NDCG = DCG/(IDCG+eps)
        assert NDCG<=1.0

        # precision
        TP = set(recommended_i.tolist()).intersection(set(gt_items.tolist()))
        precision = round(len(TP)/new_k, 4) if new_k>0 else 0.0
        recall = round(len(TP)/len(gt_items), 4) if len(gt_items)>0 else 0.0

        return NDCG, new_k, precision, recall
