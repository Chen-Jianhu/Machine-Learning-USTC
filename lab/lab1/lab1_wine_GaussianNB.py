from lab1_diabetes_GaussianNB import *

def main():
	filename = r'wine.data'
	dataset = loadFile(filename)
	print("Data len:%d"%len(dataset))
	print("Data[0]:",dataset[0]) # [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0]
	trainingSet, testingSet = splitDataset(dataset)
	print("len(trainingSet) = ", len(trainingSet), ",len(testingSet) = ",len(testingSet))
	# 按类别提取属性特征，各类别在训练集上的先验概率
	summaries, preProb = summarizeByClass(trainingSet, labelIndex=0)
	# print(summaries[1])
	# 将训练集的标签移到最后去
	for i in range(len(testingSet)):
		testingSet[i][:-1],testingSet[i][-1]  = testingSet[i][1:], testingSet[i][0]
	predictions = getPredictions(summaries, testingSet, preProb)
	# print(predictions)
	accuracy = getAccuracy(testingSet, predictions)
	print('Accuracy: %f'%accuracy)

if __name__ == '__main__':
	main()