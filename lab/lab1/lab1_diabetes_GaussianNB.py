import math
import random
from sklearn.model_selection import train_test_split

# 加载数据文件
def loadFile(filename):
	in_f = open(filename, 'r')
	dataset = in_f.readlines()
	in_f.close()
	for i in range(0, len(dataset)):
		dataset[i] = [float(x) for x in dataset[i].split(',')]
	return dataset

#　使用sklearn库函数划分训练集与测试集
def splitDataset2(dataset, splitRatio=0.67):
	dataset = [(line[:-1], line[-1]) for line in dataset] # 最后一列是标签
	# print(dataset[0],"\n",dataset[1])
	x, y = zip(*dataset) # 解压
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=splitRatio) # 67%的训练集
	# print(len(x_train),len(x_test))

def splitDataset(dataset, splitRatio=0.67):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy] # 返回训练集与测试集

# 按类别划分数据
def separateByClass(dataset, labelIndex):
	separated = {}
	counts = {}
	preProb = {} # 计算先验概率
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[labelIndex] not in separated):
			separated[vector[labelIndex]] = []
			counts[vector[labelIndex]] = 0
		counts[vector[labelIndex]] += 1
		separated[vector[labelIndex]].append(vector)
	# 计算每个类别的先验概率
	for classValue, count in counts.items():
		preProb[classValue]=counts[classValue]/len(dataset)
	return separated, preProb


def meanAndStdev(numbers):
	avg = sum(numbers)/float(len(numbers))
	variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
	return avg, math.sqrt(variance)

def summarize(dataset, labelIndex):
	summaries = [meanAndStdev(attribute) for attribute in zip(*dataset)]
	del summaries[labelIndex] # 最后的类别不用求均值和方差
	return summaries

# 按类别提取属性特征
def summarizeByClass(dataset, labelIndex):
	separated, preProb = separateByClass(dataset, labelIndex)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances, labelIndex)
	return summaries, preProb

# 计算高斯概率密度函数
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# 计算所属类的概率：合并一个数据样本中所有属性的概率，最后便得到整个数据样本属于某个类的概率
def calculateClassProbabilities(summaries, inputVector, preProb):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = preProb[classValue] # 初始化为该类别的先验概率
		for i in range(len(classSummaries)): # 对于每一个属性预测类别值
			mean, stdev = classSummaries[i]
			x = inputVector[i] # 某个属性值
			probabilities[classValue] *= calculateProbability(x, mean, stdev) # 使用乘法合并概率：属性概率连乘
	return probabilities

# 单一预测：计算一个数据样本属于每个类的概率，可以找到最大的概率值，并返回关联的类
def predict(summaries, inputVector, preProb):
	probabilities = calculateClassProbabilities(summaries, inputVector, preProb)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

# 多重预测：通过对测试数据集中每个数据样本的预测，我们可以评估模型精度
# 返回每个测试样本的预测列表
def getPredictions(summaries, testSet, preProb):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i], preProb)
		predictions.append(result)
	return predictions

# 计算精度
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	filename = r'diabetes.csv'
	dataset = loadFile(filename)
	print("Data len:%d"%len(dataset))
	# print(dataset[0]) # [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0]
	trainingSet, testingSet = splitDataset(dataset)
	# 按类别提取属性特征，各类别在训练集上的先验概率
	summaries, preProb = summarizeByClass(trainingSet, labelIndex=-1)
	predictions = getPredictions(summaries, testingSet, preProb)
	accuracy = getAccuracy(testingSet, predictions)
	print('Accuracy: %f'%accuracy)

if __name__ == '__main__':
	main()


# print(dataset[0],"\n",dataset[1])
# print('Loaded data file %s with %d rows'%(filename, len(dataset)))

# 测试separateByClass函数
# dataset = [[1,20,1], [2,21,0], [3,22,1]]
# separated = separateByClass(dataset)
# print('Separated instances: ', separated)

# 测试mean_stdev函数
# numbers = [1,2,3,4,5]
# print(meanAndStdev(numbers))

# 测试summarize函数
# dataset = [[1,20,0], [2,21,1], [3,22,0]]
# summary = summarize(dataset)
# print(summary)

# 测试summarizeByClass函数
# dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
# summary = summarizeByClass(dataset)
# print(summary)

# 测试calculateProbability函数
# x = 71.5
# mean = 73
# stdev = 6.2
# probability = calculateProbability(x, mean, stdev)
# print(probability)

# 测试calculateClassProbabilities函数
# summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
# inputVector = [1.1, '?']
# probabilities = calculateClassProbabilities(summaries, inputVector)
# print(probabilities)

# 测试predict函数
# summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
# inputVector = [1.1, '?']
# result = predict(summaries, inputVector)
# print(result)

# 测试getPredictions函数
# summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
# testSet = [[1.1, '?'], [19.1, '?']]
# predictions = getPredictions(summaries, testSet)
# print(predictions)

# 测试getAccuracy函数
# testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# predictions = ['a', 'a', 'a']
# accuracy = getAccuracy(testSet, predictions)
# print(accuracy)

