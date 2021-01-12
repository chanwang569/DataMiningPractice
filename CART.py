# -*- coding: utf-8 -*-

from math import log
import operator
import treePlotter

def splitDataSet(dataSet, axis, value):
	"""
	输入：数据集，选择维度，选择值
	输出：划分数据集
	描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
	"""
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reduceFeatVec = featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	"""
	输入：数据集
	输出：最好的划分维度
	描述：选择最好的数据集划分维度
	"""
	numFeatures = len(dataSet[0]) - 1
	bestGini = 999999.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		gini = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			subProb = len(splitDataSet(subDataSet, -1, 0)) / float(len(subDataSet))
			gini += prob * (1.0 - pow(subProb, 2) - pow(1 - subProb, 2))
		if (gini < bestGini):
			bestGini = gini
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	"""
	输入：分类类别列表
	输出：子节点的分类
	描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
		  采用多数判决的方法决定该子节点的分类
	"""
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key=lambda x: x[1])
	return sortedClassCount[0][0]

def TrainCART(dataSet, labels):
	"""
	输入：数据集，特征标签
	输出：决策树
	描述：递归构建决策树，利用上述的函数
	"""
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		# 类别完全相同，停止划分
		return classList[0]
	if len(dataSet[0]) == 1:
		# 遍历完所有特征时返回出现次数最多的
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	# 得到列表包括节点所有的属性值
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = TrainCART(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree

def classify(inputTree, featLabels, testVec):
	"""
	输入：决策树，分类标签，测试数据
	输出：决策结果
	描述：跑决策树
	"""
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	classLabel = 1
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

def ClassifyAll(inputTree, featLabels, testDataSet):
	
	classLabelAll = []
	for testVec in testDataSet:
		classLabelAll.append(classify(inputTree, featLabels, testVec))
	return classLabelAll

def Persist(inputTree, filename):
	
	import pickle
	fw = open(filename, 'wb')
	pickle.dump(inputTree, fw)
	fw.close()

def LoadModel(filename):
	import pickle
	fr = open(filename, 'rb')
	return pickle.load(fr)