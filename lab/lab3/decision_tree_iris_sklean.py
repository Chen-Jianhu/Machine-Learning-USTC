# -*- coding: utf-8 -*-
# @File 	: decision_tree_iris_sklean.py
# @Author 	: jianhuChen
# @Date 	: 2018-12-29 15:39:45
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-05 15:09:28

from sklearn import datasets
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 画图
from sklearn.externals.six import StringIO
import pydotplus

def main():
	# 得到数据集
	iris = datasets.load_iris()
	iris_feature, iris_target = iris.data, iris.target
	
	# 划分数据集
	feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33, shuffle=True)

	dt_model = tree.DecisionTreeClassifier(criterion='entropy') # 所以参数均置为默认状态
	dt_model.fit(feature_train,target_train) # 使用训练集训练模型

	predict_results = dt_model.predict(feature_test) # 使用模型对测试集进行预测

	accuracy = accuracy_score(predict_results, target_test) # 计算预测结果的准确度
	# 在 scikit-learn 中的分类决策树模型就带有 score 方法，只是传入的参数和 accuracy_score() 不太一致
	# scores = dt_model.score(feature_test, target_test)
	print('accuracy = ', accuracy)
	
	# 决策树可视化
	dot_data = StringIO()
	tree.export_graphviz(dt_model,
                        out_file=dot_data,
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True,rounded=True,
                        impurity=True)

	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	# 输出pdf，显示整个决策树的思维过程
	graph.write_pdf("iris_tree.pdf")

if __name__ == '__main__':
	main()


# DecisionTreeClassifier() 型方法中也包含非常多的参数值。例如：
# criterion = gini/entropy 可以用来选择用基尼指数或者熵来做损失函数。
# splitter = best/random 用来确定每个节点的分裂策略。支持“最佳”或者“随机”。
# max_depth = int 用来控制决策树的最大深度，防止模型出现过拟合。
# min_samples_leaf = int 用来设置叶节点上的最少样本数量，用于对树进行修剪。


# 参考链接：https://blog.csdn.net/oxuzhenyi/article/details/76427704