# iris.py
# 实现KNN算法
# by jhchen

import numpy as np
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split # 用于从数据中划分测试集

# 得到数据集
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

# 定义分类器并传入训练集
knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 计算正确率 
accuracy = np.sum(y_test == y_pred)/len(y_test)

print("accuracy = ", accuracy)
