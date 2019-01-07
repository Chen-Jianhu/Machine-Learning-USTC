## 安装

- 类UNIX操作系统

  在`terminal`中`cd`到`libsvm-3.22/python`目录下，执行`make`

- Windows

  For windows, the shared library libsvm.dll for 32-bit python is ready in the directory `..\windows`. You can also copy it to the system directory (e.g., `C:\WINDOWS\system32\` for Windows XP). To regenerate
  the shared library, please follow the instruction of building windows binaries in LIBSVM README.（摘自README文档）



## 简单介绍

libsvm库在python平台有有两个库文件：

- `svm.py`
- `svmutil.py`

他们分别对应于底层接口和高层接口

`svm.py`直接调用了c的接口，所有参数和返回值都是 `ctypes` 格式，所以小小心处理

我这次试验就直接使用`svmutil.py`这个库啦，它与上次实验所用的`LIBSVM MATLAB`的接口使用方法类似



## 线性SVM分类器

将数据集`ex7Data.zip`解压后放到`libsvm-3.22/python`目录下

> 目录结构如下：
>
> ```bash
> libsvm-3.22
> ├── python # 里面包含python要用的svm库文件
> │   ├── ex7Data # 数据集
> │   │   ├── email_test.txt
> │   │   ├── email_train-100.txt
> │   │   ├── email_train-400.txt
> │   │   ├── email_train-50.txt
> │   │   ├── email_train-all.txt
> │   │   └── twofeature.txt
> │   ├── svm.py
> │   ├── svmutil.py
> │   └── ...# 其他文件
> └── ... # 其他平台的库文件
> ```



### 实验代码

```python
# -*- coding: utf-8 -*-
# @File 	: svm_ex7.py
# @Author 	: jianhuChen
# @Date 	: 2019-01-07 12:23:24
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-07 12:30:21

from svmutil import *

train_y, train_x = svm_read_problem('ex7Data/email_train-all.txt')
test_y, test_x = svm_read_problem('ex7Data/email_test.txt')

print(len(train_y), len(train_y))
print(len(test_y), len(test_x))

model = svm_train(train_y, train_x, '-c 4')

p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
```

> '-c 4'表示`cost`的选择
>
> 具体参数表示的意义：（摘自库文件`svmuti.pyl`）
>
> ```python
> '''
> options:
> 	    -s svm_type : set type of SVM (default 0)
> 	        0 -- C-SVC		(multi-class classification)
> 	        1 -- nu-SVC		(multi-class classification)
> 	        2 -- one-class SVM
> 	        3 -- epsilon-SVR	(regression)
> 	        4 -- nu-SVR		(regression)
> 	    -t kernel_type : set type of kernel function (default 2)
> 	        0 -- linear: u'*v
> 	        1 -- polynomial: (gamma*u'*v + coef0)^degree
> 	        2 -- radial basis function: exp(-gamma*|u-v|^2)
> 	        3 -- sigmoid: tanh(gamma*u'*v + coef0)
> 	        4 -- precomputed kernel (kernel values in training_set_file)
> 	    -d degree : set degree in kernel function (default 3)
> 	    -g gamma : set gamma in kernel function (default 1/num_features)
> 	    -r coef0 : set coef0 in kernel function (default 0)
> 	    -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
> 	    -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
> 	    -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
> 	    -m cachesize : set cache memory size in MB (default 100)
> 	    -e epsilon : set tolerance of termination criterion (default 0.001)
> 	    -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
> 	    -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
> 	    -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
> 	    -v n: n-fold cross validation mode
> 	    -q : quiet mode (no outputs)
> '''
> ```

### 运行结果

```python
700 700
260 260
*
optimization finished, #iter = 301
nu = 0.321594
obj = -595.251186, rho = -0.134909
nSV = 289, nBSV = 187
Total nSV = 289
Accuracy = 98.4615% (256/260) (classification)
```



## 非线性SVM分类器

> 使用高斯核函数

- 将数据集`ex8Data.zip`解压后放到`libsvm-3.22/python`目录下

>  目录结构如下：
>
>  ```bash
>  libsvm-3.22
>  ├── matlab
>  │   ├── ex7Data
>  │   │   ├── email_test.txt
>  │   │   ├── email_train-100.txt
>  │   │   ├── email_train-400.txt
>  │   │   ├── email_train-50.txt
>  │   │   ├── email_train-all.txt
>  │   │   └── twofeature.txt
>  │   ├── ex8Data # 这次试验要用的数据集
>  │   │   ├── ex8a.txt
>  │   │   └── ex8b.txt
>  │   ├── svm.py
>  │   ├── svmutil.py
>  │   └── ... # 其他文件
>  └── ... # 其他平台的库文件
>  ```




### 实验代码

```python
# -*- coding: utf-8 -*-
# @File 	: svm_ex8.py
# @Author 	: jianhuChen
# @Date 	: 2019-01-07 12:30:07
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-07 12:52:11

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

# 设置参数：多分类+高斯卷积核 gamma=650
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
```



### 运行结果

```python
==================== data:ex8a ====================
.*.*
optimization finished, #iter = 1214
nu = 0.155791
obj = -44.567434, rho = -0.145301
nSV = 241, nBSV = 5
Total nSV = 241
Accuracy = 76.8707% (226/294) (classification)
==================== data:ex8b ====================
.*..*
optimization finished, #iter = 477
nu = 0.356369
obj = -31.767381, rho = -0.020226
nSV = 102, nBSV = 19
Total nSV = 102
Accuracy = 84.7222% (61/72) (classification)
[Finished in 0.1s]
```



## Iris 数据集

- 先写一个脚本把sklearn中的iris数据集转换成`libsvm库`所要求的格式

```python
# -*- coding: utf-8 -*-
# @File 	: convert_iris.py
# @Author 	: jianhuChen
# @Date 	: 2019-01-06 11:16:30
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-06 11:41:47

import random
from sklearn import datasets

def main():
	# 得到数据集
	iris = datasets.load_iris()
	iris_feature, iris_target = iris.data, iris.target
	# print(iris_target)
	
	fo_train = open("iris_data_train.txt", 'w')
	fo_test = open("iris_data_test.txt", 'w')

	for i in range(len(iris_feature)):
		target = iris_target[i]
		feature = ''
		for j in range( len(iris_feature[0])):
			feature += str(j+1) + ':' + str(iris_feature[i][j]) + ' '
		line = str(target) + ' ' +feature[:-1] + '\n'
		if random.randint(0, 10) < 3:
			fo_test.write(line)
		else:
			fo_train.write(line)

	fo_train.close()
	fo_test.close()

if __name__ == '__main__':
	main()
```

> 上面的脚本会生成两个文件：
>
> 训练集：iris_data_train.txt
>
> 测试集：iris_data_test.txt
>
> 内容格式如下：
>
> ```python
> 0 1:5.0 2:3.6 3:1.4 4:0.2
> 0 1:4.6 2:3.4 3:1.4 4:0.3
> 0 1:4.4 2:2.9 3:1.4 4:0.2
> ......
> ```
>
>

- 再使用`python`调用`svmutil.py库`来做分类


### 实验代码

```python
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
```



### 运行结果

```python
==================== data:iris 线性svm分类器 ====================
*
optimization finished, #iter = 29
nu = 0.016086
obj = -2.477452, rho = 0.070837
nSV = 7, nBSV = 0
*
optimization finished, #iter = 32
nu = 0.013453
obj = -1.910155, rho = 0.145326
nSV = 9, nBSV = 0
*
optimization finished, #iter = 49
nu = 0.177664
obj = -37.631634, rho = 0.285131
nSV = 16, nBSV = 9
Total nSV = 26
Accuracy = 97.8261% (45/46) (classification)
==================== data:iris 非线性svm分类器 ====================
*.*
optimization finished, #iter = 81
nu = 0.114690
obj = -4.415275, rho = 0.160565
nSV = 26, nBSV = 0
*
optimization finished, #iter = 69
nu = 0.141796
obj = -5.047469, rho = 0.329249
nSV = 26, nBSV = 1
*.*
optimization finished, #iter = 75
nu = 0.275659
obj = -12.757217, rho = 0.216891
nSV = 33, nBSV = 9
Total nSV = 48
Accuracy = 100% (46/46) (classification)
[Finished in 0.1s]
```



















