# -*- coding: utf-8 -*-
# @File     : decision_tree.py
# @Author   : jianhuChen
# @Date     : 2018-12-29 12:22:14
# @License  : Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-05 16:02:11
# @Reference: https://blog.csdn.net/moxigandashu/article/details/71305273?locationNum=9&fps=1

from math import log

# 计算信息熵 H(x)
def calcShannonEnt(dataSet):
	numEntries = len(dataSet) # 样本总数
	# 为所有的分类类目创建字典（类别的名称为键，该类别的个数为值）
	labelCounts = {}
	for featVec in dataSet:
		currentLable = featVec[-1] # 取得最后一列数据
		if currentLable not in labelCounts.keys(): # 还没添加到字典里的类型
			labelCounts[currentLable] = 0
		labelCounts[currentLable] += 1
	# 统计完每个分类的样本个数后，计算香农熵
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries # 每种类型个数占所有的比值
		shannonEnt += -(prob * log(prob, 2))
	return shannonEnt # 返回熵

#　输入三个变量（待划分的数据集，特征号，分类值）
# 返回含有该特征值且去掉该特征值后的数据集
def splitDataSet(dataSet, axis, value):
	retDataSet = [] # 存储去掉该特征值之后的数据集
	for featVec in dataSet: # 去掉 dataSet矩阵中的第axis列的值等于value的样例的此属性
		if featVec[axis] == value: # 值等于value的，每一行为新的列表（去除第axis个数据）
			reduceFeatVec = featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet # 返回分类后的新矩阵

# 返回按哪个特征划分信息增益最大
def chooseBestFeatureToSplit(dataSet, criterion):
	numFeature = len(dataSet[0])-1 # 求属性的个数，最后一个为label，所以要减掉
	baseEntropy = calcShannonEnt(dataSet) # 香农熵 H(x)
	bestInforGain = 0.0 # 因为要求最大信息增益，所以先初始化为0
	bestFeature = -1 # 设使得信息增益最大的特征为-1
	for i in range(numFeature): # 对于每一列特征 ，求信息增益（比）
		featList = [number[i] for number in dataSet] # 得到某个特征下所有值（某列）
		uniqualVals = set(featList) # set：集合，特性是无重复元素，这里使用它的作用是得到该属性的所有可能值
		newEntropy = 0.0
		splitInfo = 0.0 # 训练集关于特征i的值的熵 
		for value in uniqualVals: #　对于某个特征值
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet)) # 求出该值在i列属性中的概率，即p(t)
			newEntropy += prob*calcShannonEnt(subDataSet) # 求i列属性各值对于的熵求和
			splitInfo -= prob * log(prob, 2)
		if criterion == 'ID3': # ID3算法使用信息增益作为划分标准 C4.5算法使用信息增益作为划分标准	
			infoGain = baseEntropy-newEntropy # 计算信息增益
		elif criterion == 'C4.5':
			infoGain = (baseEntropy - newEntropy) / splitInfo;  #求出第i列属性的信息增益率
		# 保存信息增益（率）最大的值以及所在的下标（列值i）
		if infoGain > bestInforGain:
			bestInforGain = infoGain
			bestFeature = i
	return bestFeature # 返回该属性所在的列号

# 投票表决代码
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
	return sortedClassCount[0][0]

# 创建决策树构造函数createTree
def createTree(dataSet, feature, criterion='ID3'):
	classList = [example[-1] for example in dataSet] # 创建需要创建树的训练数据的结果列表（例如最外层的列表是[N, N, Y, Y, Y, N, Y]）
	# 类别相同，停止划分 ,递归终点
	if classList.count(classList[-1]) == len(classList): 
		return classList[-1]

	# 训练数据只给出类别数据（没给任何属性值数据），返回出现次数最多的分类名称
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	
	# 按照信息增益最高选取分类特征属性
	bestFeat = chooseBestFeatureToSplit(dataSet, criterion) # 返回分类的特征下标
	bestFeatFeature = feature[bestFeat] # 该特征的feature
	myTree = {bestFeatFeature:{}} # 构建树的字典
	del(feature[bestFeat]) # 从feature的list中删除该feature
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals=set(featValues) # 按照该属性划分后，这个属性有多少种取值情况，当前节点就有多少分支
	for value in uniqueVals: # 根据该属性的值求树的各个分支
		subFeature=feature[:] # 子集合
		# 构建数据的子集合，并进行递归
		myTree[bestFeatFeature][value]=createTree(splitDataSet(dataSet, bestFeat, value), subFeature, criterion) # 根据各个分支递归创建树
	return myTree

# 决策树运用于分类
def classify(inputTree, feature, testVec):
	firstStr = list(inputTree.keys())[0] # 获取树的第一个特征属性
	secondDict = inputTree[firstStr] # 树的分支，子集合Dict
	featIndex = feature.index(firstStr) # 获取决策树第一层在featLables中的位置
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict': # 还有分支，继续往下递归查找
				return classify(secondDict[key],feature,testVec)
			else:
				return secondDict[key]

# 存储树
def storeTree(inputTree, filename):
	fw = open(filename, 'w')
	fw.write(str(inputTree))
	fw.close()
	print("Store complete.")

#　读入树
def loadTree(filename):
	try:
		fr = open(filename, 'r')
		tree_dict = eval(fr.readline()) # 将str转换成dict
		fr.close()
		if type(tree_dict).__name__ == 'dict':
			print("Load complete.")
			return tree_dict
		else:
			return None
	except FileNotFoundError:
		print("File is not found.")
		return None

def createDataSet():
	dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	feature=['no surfacing','flippers']
	return dataSet, feature


def main():
	myDat, feature = createDataSet()
	print(myDat, feature)
	shanVal = calcShannonEnt(myDat)
	print(shanVal)
	myTree = createTree(myDat, feature, criterion='ID3')
	print(myTree)
	storeTree(myTree, 'test_tree.dat')
	t = loadTree('test_tree.dat')
	if t != None:
		print(type(t))



if __name__ == '__main__':
	main()