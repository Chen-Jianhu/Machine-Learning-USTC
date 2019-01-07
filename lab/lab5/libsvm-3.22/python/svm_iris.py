# -*- coding: utf-8 -*-
# @File 	: svm_iris.py
# @Author 	: jianhuChen
# @Date 	: 2019-01-07 12:50:07
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-07 13:13:39

from svmutil import *

#================数据集：iris 线性svm分类器========================
print('='*20, 'data:iris 线性svm分类器', '='*20)

train_y, train_x = svm_read_problem('iris_data_train.txt')
test_y, test_x = svm_read_problem('iris_data_test.txt')
# print(train_y[:2], train_y[:2])
# 读入的数据格式：
# [0.0, 0.0] [{1: 5.1, 2: 3.5, 3: 1.4, 4: 0.2}, {1: 4.9, 2: 3.0, 3: 1.4, 4: 0.2}]

model = svm_train(train_y, train_x, '-c 4') # cost

p_label, p_acc, p_val = svm_predict(test_y, test_x, model)


#================数据集：iris 非线性svm分类器========================
print('='*20, 'data:iris 非线性svm分类器', '='*20)

train_y, train_x = svm_read_problem('iris_data_train.txt')
test_y, test_x = svm_read_problem('iris_data_test.txt')
# print(train_y[:2], train_y[:2])
# 读入的数据格式：
# [0.0, 0.0] [{1: 5.1, 2: 3.5, 3: 1.4, 4: 0.2}, {1: 4.9, 2: 3.0, 3: 1.4, 4: 0.2}]

prob  = svm_problem(train_y, train_x, isKernel=True)
# 设置参数：多分类+高斯卷积核 gamma=2
param = svm_parameter('-s 0 -t 2 -g 2')
model = svm_train(prob, param)

p_label, p_acc, p_val = svm_predict(test_y, test_x, model)