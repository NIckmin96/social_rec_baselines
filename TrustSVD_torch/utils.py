import torch
import torch.nn as nn
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
    def __init__(self, data, user_item_dict, item_user_dict, user_user_dict, user_user_in_dict):
        super().__init__()
        self.data = data
        self.user_item_dict = user_item_dict
        self.item_user_dict = item_user_dict
        self.user_user_dict = user_user_dict
        self.user_user_in_dict = user_user_in_dict
        self.user_id = data['user_id'].values
        self.product_id = data['product_id'].values
        self.rating = data['rating'].values

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_id = torch.tensor(self.user_id[idx]).long()
        product_id = torch.tensor(self.product_id[idx]).long()
        rating = torch.tensor(self.rating[idx]).float()

        i_u = torch.tensor(self.user_item_dict.get(user_id.item(),[])).long()
        t_u = torch.tensor(self.user_user_dict.get(user_id.item(),[])).long()

        len_u_j = torch.tensor(len(self.item_user_dict.get(product_id.item(),[]))).long()
        len_u_i = torch.tensor([len(self.item_user_dict.get(item.item(),[])) for item in i_u if item.item()!=0]).long()
        len_t_v = torch.tensor([len(self.user_user_in_dict.get(user.item(),[])) for user in t_u if user.item()!=0]).long()

        data = {
            'user_id': user_id,
            'product_id': product_id,
            'rating': rating,
            'i_u' : i_u,
            't_u' : t_u,
            'len_u_i' : len_u_i,
            'len_u_j' : len_u_j,
            'len_t_v' : len_t_v
        }
        # print(data['user_id'], data['product_id'], data['rating'])

        return data
