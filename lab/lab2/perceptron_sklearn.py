# -*- coding: utf-8 -*-
# @File 	: perceptron.py
# @Author 	: jianhuChen
# @Date 	: 2018-12-23 17:30:37
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2018-12-23 20:02:44

from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from collections import defaultdict
from sklearn.model_selection import train_test_split # 用于从数据中划分测试集
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

def data_preprocessing(fileName, isSave=False):
	df = pd.read_csv(fileName)
	# print(df.shape)
	X = df.drop('dataClass', axis=1)
	# print(X.shape)
	y = df.dataClass
	y = LabelBinarizer().fit_transform(y)
	# print(y)

	#映射特征字段，将字符映射为数值；都是单一类型直接映射
	d = defaultdict(LabelEncoder)
	X = X.apply(lambda x: d[x.name].fit_transform(x)) # 转换成数字编码
	
	# 归一化
	sc = StandardScaler()
	sc.fit(X)
	X = sc.transform(X)

	# 是否保存
	if isSave:
		X.to_csv("preproce"+fileName,encoding='utf-8',index=False)
	
	# 划分数据集
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
	return X_train, X_test, y_train, y_test


if __name__ == '__main__':
	X_train, X_test, y_train, y_test = data_preprocessing('adult.csv')
	print(len(X_train),len(X_test))

	# 定义分类器并传入训练集
	clf = Perceptron(n_iter = 40)
	clf.fit(X_train, y_train)
	# 打印权值矩阵
	# print(clf.coef_)
	y_pred = clf.predict(X_test)
	# 计算正确率  
	accuracy = accuracy_score(y_pred, y_test)
	print("Perceptron:", accuracy)


	# 定义分类器并传入训练集
	knn = neighbors.KNeighborsClassifier(n_neighbors=10)
	knn.fit(X_train, y_train)
	# 预测
	y_pred = knn.predict(X_test)
	accuracy = accuracy_score(y_pred, y_test)
	print("KNN", accuracy)