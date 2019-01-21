# -*- coding: utf-8 -*-
# @File 	: svm_ex8.py
# @Author 	: jianhuChen
# @Date 	: 2019-01-07 12:30:07
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-13 18:45:29

from svmutil import *

#================数据集：ex8a=============================
print('='*20, 'data:ex8a', '='*20)
y, x = svm_read_problem('ex8Data/ex8a.txt')
# print(y[:2], x[:2])
# 读入的数据格式：
# [1.0, 1.0] [{1: 0.107143, 2: 0.60307}, {1: 0.093318, 2: 0.649854}]

# 划分训练集与测试集
split_num = int(len(y)*0.66)
prob  = svm_problem(y[:split_num], x[:split_num], isKernel=True)

# 设置参数：多分类+高斯核 gamma=650
param = svm_parameter('-s 0 -t 2 -g 650')
model = svm_train(prob, param)

p_label, p_acc, p_val = svm_predict(y[split_num:], x[split_num:], model)

#================数据集：ex8b=============================
print('='*20, 'data:ex8b', '='*20)

y, x = svm_read_problem('ex8Data/ex8b.txt')
# print(y[:2], x[:2])
# 读入的数据格式：
# [1.0, 1.0] [{1: 0.107143, 2: 0.60307}, {1: 0.093318, 2: 0.649854}]

# 划分训练集与测试集
split_num = int(len(y)*0.66)
prob  = svm_problem(y[:split_num], x[:split_num], isKernel=True)

# 设置参数：多分类+高斯卷积核 gamma=120
param = svm_parameter('-s 0 -t 2 -g 120')
model = svm_train(prob, param)

p_label, p_acc, p_val = svm_predict(y[split_num:], x[split_num:], model)

