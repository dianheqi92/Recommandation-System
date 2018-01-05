import numpy as np
import pandas as pd

from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.linalg import lstsq

import time
start_time = time.time()

def base(train_data, train_data_matrix) :
	# calculate average rating
	sum_rating = 0
	line_num = 0
	for line in train_data.itertuples():
		sum_rating = sum_rating + line[3]
		line_num += 1
	avg_rating = sum_rating / line_num
	
	#update bu
	bias_user = []
	for i in range (train_data_matrix.shape[0]):
		sum_bu = 0
		valid_count = 0
		for j in range (train_data_matrix.shape[1]):
			if(train_data_matrix[i][j] != 0):
				valid_count += 1
				sum_bu += (train_data_matrix[i][j] - avg_rating)
		if (valid_count == 0):              #if user i doesn't rate any movie, bias = 0
			bias_user.append(0)
		else:
			bias_user.append(sum_bu / valid_count)
		
		#update bi
	bias_item = []
	for j in range (train_data_matrix.shape[1]):
		sum_bi = 0
		valid_count = 0
		for i in range (train_data_matrix.shape[0]):
			if(train_data_matrix[i][j] != 0):
				valid_count += 1
				sum_bi += (train_data_matrix[i][j] - avg_rating)
		if(valid_count == 0):
			bias_item.append(0)
		else:
			bias_item.append(sum_bi / valid_count)
	
	#prediction
	baseline_pred = np.zeros(train_data_matrix.shape)
	base_diff = np.zeros(train_data_matrix.shape)
	for i in range (n_users):
		for j in range (n_items):
			baseline_pred[i][j] = avg_rating + bias_user[i] + bias_item[j]
			if (baseline_pred[i][j] > 5):
				baseline_pred[i][j] = 5
			elif (baseline_pred[i][j] < 1):
				baseline_pred[i][j] = 1
			if (train_data_matrix[i][j] != 0) :
				base_diff[i][j] = train_data_matrix[i][j] - baseline_pred[i][j]
	return baseline_pred, base_diff


def low_rank(train_data_matrix, k_num, loop) :
	# get SVD components from train matrix. Choose k.
	u, s, vt = svds(train_data_matrix, k=k_num)
	s_diag_matrix=np.diag(s)
	P = np.dot(u, s_diag_matrix)
	Q = vt
	P_pre = np.zeros(P.shape)
	Q_pre = np.zeros(Q.shape)
	m = 0
	P_zeros = np.zeros(P.shape).astype(np.int)
	Q_zeros = np.zeros(Q.shape).astype(np.int)
	P_diff = np.ones(P.shape)
	Q_diff = np.ones(Q.shape)
	while True:
		if (np.array_equal(P_zeros,P_diff) & np.array_equal(Q_zeros,Q_diff)) :
			break
		m = m+1
		if (m > loop):
			break;
		#print (m)
		P_diff = (np.subtract(P, P_pre)*100000000).astype(np.int)
		Q_diff = (np.subtract(Q, Q_pre)*100000000).astype(np.int)
		P_pre = P
		Q_pre = Q
		Q = lstsq(P, train_data_matrix)[0]
		P = lstsq(Q.T, train_data_matrix.T)[0].T
	X_pred = np.dot(P, Q)
	return X_pred

def neighborhood (train_matrix,  type = 'user') :
	if type == 'user':
		user_similarity = cosine_similarity(train_data_matrix)
		user_norm = np.array([np.abs(user_similarity).sum(axis=1)])
		for i in range (user_norm.shape[1]):
			if user_norm[0][i] == 0 :
				user_norm[0][i] = 1
		pred = user_similarity.dot(train_matrix) / user_norm.T
	elif type == 'item':
		item_similarity = cosine_similarity(train_data_matrix.T)
		item_norm = np.array([np.abs(item_similarity).sum(axis=1)])
		for i in range (item_norm.shape[1]):
			if item_norm[0][i] == 0 :
				item_norm[0][i] = 1
		pred = train_matrix.dot(item_similarity) / item_norm
	return pred

	
def rmse(prediction, ground_truth):
	prediction = prediction[ground_truth.nonzero()].flatten()
	ground_truth = ground_truth[ground_truth.nonzero()].flatten()
	k = 0
	m = prediction.shape[0]
	for i in range(m):
			if ((prediction[i] < 1) |(prediction[i] > 5)) :
				# print (prediction[i])
				k = k+1
	print ('Total number is ', m)
	print ('viola number is ', k)
	return sqrt(mean_squared_error(prediction, ground_truth))

	
def norm (matrix) :
	for i in range (n_users):
		for j in range (n_items):
			if (matrix[i][j] > 5):
				matrix[i][j] = 5
			elif (matrix[i][j] < 1):
				matrix[i][j] = 1
	return matrix

def combine (matrix_1, matrix_2, baseline_pred, train_data_matrix) :
	rm_th = rmse(baseline_pred,train_data_matrix)
	for i in range (100+1):
		i = i/100
		print (i) 
		matrix = matrix_1 * i + matrix_2 * (1-i) + baseline_pred
		matrix = norm (matrix)
		rm = rmse(matrix,train_data_matrix)
		if (rm < rm_th) :
			o = matrix
			rm_th = rm
			o_i = i
	print ('we want ',o_i)
	return o
		
header = ['user_id', 'item_id', 'rating', 'timestamp']
header_test = ['user_id', 'item_id', 'timestamp']
data_set = pd.read_csv('D:\\Courses\\EE412 Foundation of Big Data Analytic\\project2\\ee412.train', sep='\t', names=header)
test_data = pd.read_csv('D:\\Courses\\EE412 Foundation of Big Data Analytic\\project2\\ee412.testset.csv', sep=',', names=header_test)
n_users = 943
n_items = 1682
# print ('Number of users = ' + str(n_users))
# print ('Number of items = ' + str(n_items))

# default ratio of seperation is 4:1
train_data, validation_data = cv.train_test_split(data_set, train_size = 0.9)

train_data_matrix = np.zeros((n_users, n_items))
validation_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
	train_data_matrix[line[1]-1, line[2]-1] = line[3]
for line in validation_data.itertuples():
	validation_data_matrix[line[1]-1, line[2]-1] = line[3]

baseline_pred, base_diff = base(train_data, train_data_matrix)
print ('Baseline train_RMSE: ' + str(rmse(baseline_pred, train_data_matrix)))
print ('Baseline validation_RMSE: ' + str(rmse(baseline_pred, validation_data_matrix)))

low_rank_diff = low_rank(base_diff, 20, 100)
low_rank_pred = baseline_pred + low_rank_diff
low_rank_pred = norm (low_rank_pred)
print ('Low_rank train_RMSE: ' + str(rmse(low_rank_pred, train_data_matrix)))
print ('Low_rank validation_RMSE: ' + str(rmse(low_rank_pred, validation_data_matrix)))

neghbor_user_diff = neighborhood (low_rank_diff,  type = 'user')
neghbor_item_diff = neighborhood (low_rank_diff,  type = 'item')
user_pred = baseline_pred + neghbor_user_diff
item_pred = baseline_pred + neghbor_item_diff
user_pred = norm (user_pred)
item_pred = norm (item_pred)
print ('User-based train_RMSE: ' + str(rmse(user_pred, train_data_matrix)))
print ('User-based validation_RMSE: ' + str(rmse(user_pred, validation_data_matrix)))
print ('Item-based train_RMSE: ' + str(rmse(item_pred, train_data_matrix)))
print ('Item-based validation_RMSE: ' + str(rmse(item_pred, validation_data_matrix)))

combination = combine(low_rank_diff, neghbor_item_diff, baseline_pred, train_data_matrix)
print ('Combination train_RMSE: ' + str(rmse(combination, train_data_matrix)))
print ('Combination validation_RMSE: ' + str(rmse(combination, validation_data_matrix)))
test_pred = []
combination = np.around(combination*2)/2
print (combination)
for line in test_data.itertuples():
	test_pred.append (combination[line[1]-1][line[2]-1])
test_data = test_data.assign(rating = test_pred)
test_data = test_data.drop(['timestamp'],axis = 1)
test_data.to_csv('D:\\Courses\\EE412 Foundation of Big Data Analytic\\project2\\test_pred_result.csv', index = False, header=False)