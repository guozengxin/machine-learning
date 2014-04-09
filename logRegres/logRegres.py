#!/usr/bin/env python
#encoding=utf8

from numpy import mat,ones,shape,exp,array
import random

def loadDataSet():
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat

def sigmoid(inX):
	return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	m, n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights

def stocGradAscent0(dataMatrix, classLabels):
	m,n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4/(1.0+j+i) + 0.01
			randIndex = int(random.uniform(0, len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights

def classifyVector(inX, weights):
	prob = sigmoid(sum(inX*weights))
	if prob > 0.5: return 1.0
	else: return 0.0

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []
	trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
	trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
	errorCount = 0
	numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print "the error rate of this test is: %f" % errorRate
	return errorRate

def multiTest():
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))

def test1():
	dataArr, labelMat = loadDataSet()
	weights = gradAscent(dataArr, labelMat)
	print weights

def test2():
	dataArr, labelMat = loadDataSet()
	weights = stocGradAscent0(array(dataArr), labelMat)
	print weights

def test3():
	dataArr, labelMat = loadDataSet()
	weights = stocGradAscent1(array(dataArr), labelMat)
	print weights

def test4():
	multiTest()

if __name__ == '__main__':
	test4()
