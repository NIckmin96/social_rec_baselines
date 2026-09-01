import numpy as np

def precision_k(score_label, k):
    p, i = 0, 0
    for s in score_label:
        if i < k:
            if s[1] > 3:
                p += 1
            i += 1
    return p/k

def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            # dcg += (2**s[1]-1) / np.log2(2+i) # org
            dcg += s[1] / np.log2(2+i)
            i += 1
    return dcg

def ndcg_k(score_label, k):
    score_label_ = sorted(score_label, key=lambda d:d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            # norm += (2**s[1]-1) / np.log2(2+i)
            norm += s[1] / np.log2(2+i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm

def auc(score_label):
    fp1, tp1, fp2, tp2, auc = 0.0, 0.0, 0.0, 0.0, 0.0
    for s in score_label:
        fp2 += (1-s[1]) # noclick
        tp2 += s[1] # click
        auc += (tp2 - tp1) * (fp2 + fp1) / 2
        fp1, tp1 = fp2, tp2
    try:
        return 1- auc / (tp2 * fp2)
    except:
        return 0.5

def mae(score_label):
    n = 0
    error = 0
    for s in score_label:
        if s[1] == 0:
            continue
        error += abs(s[1] - s[0])
        n += 1
    return error / n if n else 0

def rmse(score_label):
    n = 0
    error = 0
    for s in score_label:
        if s[1] == 0:
            continue
        error += (s[1] - s[0]) ** 2
        n += 1
    return np.sqrt(error/n) if n else 0

def get_ndcg(score_label, k=10):
    eps = 1e-10
    if not score_label:
        return 0.0, 0, 0.0, 0.0
    score_label = np.asarray(score_label)
    logits = score_label[:, 0]
    ratings = score_label[:, 1]
    items = np.arange(len(ratings))
    new_k = min(k, len(score_label))
    if new_k == 0:
        return 0.0, 0, 0.0, 0.0

    rec_indices = np.argsort(-logits)[:new_k]
    recommended_i = items[rec_indices]
    recommended_r = ratings[rec_indices]

    ideal_indices = np.argsort(-ratings)[:new_k]
    ideal_r = ratings[ideal_indices]

    discount = np.log2(np.arange(new_k) + 2)
    dcg = np.sum(recommended_r / discount)
    idcg = np.sum(ideal_r / discount)
    ndcg = dcg / (idcg + eps)
    # ndcg *= (new_k/k) # 보정

    gt_items = set(items[ratings != 0].tolist())
    if gt_items:
        tp = set(recommended_i.tolist()).intersection(gt_items)
        precision = round(len(tp) / new_k, 4)
        recall = round(len(tp) / len(gt_items), 4)
    else:
        precision = 0.0
        recall = 0.0

    return ndcg, new_k, precision, recall
