import os
import time
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import sys
import csv
import eval
from input import DataInput
from model import Model

learning_rate = 0.1
keep_prob = 0.5
lambda1 = 0.001
lambda2 = 0.001
trunc_len = 10
train_batch_size = 64
test_batch_size = 64
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regen', action='store_true')
    parser.add_argument('--neg_train', action='store_true',
                        help='Enable negative sampling for train.')
    parser.add_argument('--no_neg_test', action='store_true',
                        help='Disable negative sampling for test.')
    args = parser.parse_args()
    return args

args = get_args()
random.seed(args.seed)
np.random.seed(args.seed)
tf.compat.v1.set_random_seed(args.seed) 

if not (os.path.isfile(f'data/{args.dataset}/dataset_{args.seed}.pkl') & os.path.isfile(f'data/{args.dataset}/list_{args.seed}.pkl')) or args.regen:
	from build_dataset import create_dataset
	create_dataset(args)

with open(f'data/{args.dataset}/dataset_{args.seed}.pkl', 'rb') as f:
	train_set = pickle.load(f)
	test_set = pickle.load(f)
with open(f'data/{args.dataset}/list_{args.seed}.pkl', 'rb') as f:
	u_friend_list = pickle.load(f)
	u_read_list = pickle.load(f)
	uf_read_list = pickle.load(f) 
	i_friend_list = pickle.load(f)
	i_read_list = pickle.load(f)
	if_read_list = pickle.load(f)
	i_link_list = pickle.load(f)
	user_count, item_count = pickle.load(f)
	

def calc_metric(score_label_u):
	score_label_u = sorted(score_label_u, key=lambda d:d[0], reverse=True)
	# precision = np.array([eval.precision_k(score_label_u, k) for k in range(1, 21)])
	# ndcg = np.array([eval.ndcg_k(score_label_u, k) for k in range(1, 21)])
	# ndcg = eval.ndcg_k(score_label_u, 10)
	precision = []
	ndcg = []
	for k in range(1, 21):
		ndcg_k, _, precision_k, _ = eval.get_ndcg(score_label_u, k)
		ndcg.append(ndcg_k)
		precision.append(precision_k)
	precision = np.array(precision)
	ndcg = np.array(ndcg)
	auc = eval.auc(score_label_u)
	mae = eval.mae(score_label_u)
	rmse = eval.rmse(score_label_u)
	return precision, ndcg, auc, mae, rmse

def get_metric(score_label):
	Precision = np.zeros(20)
	NDCG = np.zeros(20)
	AUC = 0.
	score_df = pd.DataFrame(score_label, columns=['uid', 'score', 'label'])
	num = 0
	score_label_all = []
	for uid, hist in score_df.groupby('uid'):
		if hist.shape[0]<10:
			continue
		score = hist['score'].tolist()
		label = hist['label'].tolist()
		score_label_u = []
		for i in range(len(score)):
			score_label_u.append([score[i], label[i]])
			score_label_all.append([score[i], label[i]])
		precision, ndcg, auc, mae, rmse = calc_metric(score_label_u)
		Precision += precision
		NDCG += ndcg
		AUC += auc
		num += 1
	# ==== NEGATIVE SAMPLE DIAGNOSTICS START ====
	if score_label_all:
		neg_count = sum(1 for _, label in score_label_all if label == 0)
		total_count = len(score_label_all)
		print(f"[NEG DIAG] score_label_all total={total_count} neg={neg_count} pos={total_count - neg_count}")
	else:
		print("[NEG DIAG] score_label_all is empty")
	# ==== NEGATIVE SAMPLE DIAGNOSTICS END ====
	score_label_all = sorted(score_label_all, key=lambda d:d[0], reverse=True)
	GPrecision = np.array([eval.precision_k(score_label_all, k*len(score_label_all)/100) for k in range(1, 21)])
	GAUC = eval.auc(score_label_all)
	MAE = eval.mae(score_label_all)
	RMSE = eval.rmse(score_label_all)
	return Precision / num, NDCG / num, AUC / num, GPrecision, GAUC, MAE, RMSE
		
def _eval(sess, model):
	loss_sum = 0.
	pos_sq_sum = 0.0
	pos_count = 0
	batch = 0
	score_label = []
	for _, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, \
		i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput in \
	DataInput(test_set, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list, if_read_list, \
		i_link_list, test_batch_size, trunc_len):
		score_, loss = model.eval(sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, \
		u_friend_l, uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput, lambda1, lambda2)
		for i in range(len(score_)):
			label = datainput[2][i]
			if label != 0:
				err = score_[i] - label
				pos_sq_sum += err * err
				pos_count += 1
		for i in range(len(score_)):
			score_label.append([datainput[1][i], score_[i], datainput[2][i]])
		loss_sum += loss
		batch += 1
	Precision, NDCG, AUC, GPrecision, GAUC, MAE, RMSE = get_metric(score_label) 
	pos_mse = (pos_sq_sum / pos_count) if pos_count else 0.0
	return pos_mse, Precision, NDCG, MAE, RMSE


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
print("GPUs:", tf.config.list_physical_devices("GPU"))
with tf.compat.v1.Session() as sess:
	model = Model(user_count, item_count)
	sess.run(tf.compat.v1.global_variables_initializer())
	sess.run(tf.compat.v1.local_variables_initializer()) 

	sys.stdout.flush()
	lr = learning_rate
	Train_loss_pre = 100
	best_rmse = 9999
	best_mae = 9999
	cnt = 0
	early_stop = False
	epoch_size = round(len(train_set) / train_batch_size)
	print("epoch_size :", epoch_size)
	for _ in range(epoch_size):
		random.shuffle(train_set)
		iter_num, loss_sum= 0, 0.
		for _, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, \
		i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput in \
	DataInput(train_set, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list, if_read_list, \
		i_link_list, train_batch_size, trunc_len):
			loss = model.train(sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, \
			uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput, lr, keep_prob, lambda1, lambda2)
			iter_num += 1
			loss_sum += loss

			if model.global_step.eval() % 1000 == 0:
				Train_loss = loss_sum / iter_num
				Test_loss, P, N, MAE, RMSE = _eval(sess, model)
				Train_loss = loss_sum / iter_num
				print('Epoch %d Step %d Train_loss: %.4f Test_loss: %.4f P@3: %.4f P@5: %.4f P@10: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f MAE: %.4f RMSE: %.4f Best_MAE: %.4f' %
				(model.global_epoch_step.eval(), model.global_step.eval(), Train_loss, Test_loss, P[2], P[4], P[9], N[2], N[4], N[9], MAE, RMSE, best_rmse))
				iter_num = 0
				loss_sum = 0.0

				if RMSE < best_rmse:
					print("Best Model Saved!")
					model.save(sess, f'model/{args.dataset}/DUAL_GAT_{args.seed}.ckpt') 
					best_rmse = RMSE
					best_mae = MAE
					cnt = 0
				elif RMSE > best_rmse:
					cnt += 1
					if cnt >= 5:
						print("Early stopping triggered.")
						early_stop = True
						break
     
				model.policy_update(sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, \
				i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput, lr/10, keep_prob, lambda1, lambda2)
    
		if early_stop:
			break

		sys.stdout.flush()
		model.global_epoch_step_op.eval()
	

	sys.stdout.flush()
	
