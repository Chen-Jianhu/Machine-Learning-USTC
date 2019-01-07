# -*- coding: utf-8 -*-
# @File 	: svm_ex7.py
# @Author 	: jianhuChen
# @Date 	: 2019-01-07 12:23:24
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-07 13:04:17

from svmutil import *

train_y, train_x = svm_read_problem('ex7Data/email_train-all.txt')
test_y, test_x = svm_read_problem('ex7Data/email_test.txt')

print(len(train_y), len(train_y))
print(len(test_y), len(test_x))

model = svm_train(train_y, train_x, '-c 4')

p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
