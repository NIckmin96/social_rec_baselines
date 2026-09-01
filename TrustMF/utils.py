import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
  
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
    
class MyDataset(Dataset):
    """Rating-level dataset for the reconstruction term (Eq. 4/5, sum over Omega):
    one sample per observed (user, item) rating."""
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.user_id = data['user_id'].values
        self.product_id = data['product_id'].values
        self.rating = data['rating'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            'user_id':torch.tensor(self.user_id[idx]).long(),
            'product_id':torch.tensor(self.product_id[idx]).long(),
            'rating':torch.tensor(self.rating[idx]).float(),
        }

        return item


class EdgeDataset(Dataset):
    """Trust-edge dataset for the trust-factorisation term (Eq. 4/5, sum over Psi):
    one sample per observed edge, so every edge contributes exactly once per epoch,
    independent of how many ratings its endpoints have."""
    def __init__(self, edges):
        super().__init__()
        self.edges = torch.as_tensor(edges, dtype=torch.long)

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        return self.edges[idx, 0], self.edges[idx, 1]