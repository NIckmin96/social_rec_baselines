import numpy as np
import torch
import torch.nn.functional as F


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items): # legacy helper (single-gt list)
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def compute_rmse_mae_ndcg(total_user, total_item, total_pred, total_rating, top_k=10):
	# scale preds to 0~5 for RMSE/MAE
	total_pred = (total_pred - total_pred.min()) / (total_pred.max() - total_pred.min() + 1e-9) * 5
	r_mask = (total_rating != 0)
	rmse = torch.sqrt(F.mse_loss(total_pred[r_mask], total_rating[r_mask]))
	mae = F.l1_loss(total_pred[r_mask], total_rating[r_mask])

	discount = torch.reciprocal(torch.log2(torch.arange(top_k, device=total_pred.device) + 2))
	dcg_total = torch.tensor(0.0, device=total_pred.device)
	idcg_total = torch.tensor(0.0, device=total_pred.device)
	precision_total = 0.0
	for user in total_user.unique():
		u_idx = (total_user == user).nonzero(as_tuple=True)[0]
		if u_idx.numel() == 0:
			continue
		user_pred = total_pred[u_idx]
		user_rating = total_rating[u_idx]
		topk = min(top_k, user_pred.numel())
		_, dcg_indices = user_pred.topk(topk)
		dcg_total += (user_rating[dcg_indices] * discount[:topk]).sum()
		hits = (user_rating[dcg_indices] > 0).float().sum()
		precision_total += (hits / topk).item()
		ideal_topk = min(top_k, user_rating.numel())
		sorted_rating, _ = user_rating.topk(ideal_topk)
		idcg_total += (sorted_rating * discount[:ideal_topk]).sum()
	ndcg = (dcg_total / (idcg_total + 1e-8)).item()
	precision = precision_total / len(total_user.unique()) if len(total_user.unique()) > 0 else 0.0
	return rmse.item(), mae.item(), ndcg, precision


def metrics(model, test_loader, top_k):
	# Collect predictions/ratings for RMSE, MAE, NDCG@top_k
	model.eval()
	u_lst, i_lst, p_lst, r_lst = [], [], [], []
	with torch.no_grad():
		for user, item_i, rating in test_loader:
			user = user.long().cuda()
			item_i = item_i.long().cuda()
			rating = rating.float().cuda()
			prediction = model(user, item_i)
			u_lst.append(user); i_lst.append(item_i); r_lst.append(rating); p_lst.append(prediction)
	total_user = torch.cat(u_lst, dim=0)
	total_item = torch.cat(i_lst, dim=0)
	total_rating = torch.cat(r_lst, dim=0)
	total_pred = torch.cat(p_lst, dim=0)
	return compute_rmse_mae_ndcg(total_user, total_item, total_pred, total_rating, top_k)
