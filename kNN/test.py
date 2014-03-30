#!/usr/bin/env python
#encoding=utf-8

import kNN
import numpy
import unittest
import os

def createDataSet():
	group = numpy.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def file2matrix(filename):
	love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)            #get the number of lines in the file
	returnMat = numpy.zeros((numberOfLines,3))        #prepare matrix to return
	classLabelVector = []                       #prepare labels return
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		if(listFromLine[-1].isdigit()):
			classLabelVector.append(int(listFromLine[-1]))
		else:
			classLabelVector.append(love_dictionary.get(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

class KNNTestCase(unittest.TestCase):
	def setUp(self):
		pass

	def testKNN(self):
		group, labels = createDataSet()
		c = kNN.classify0([0, 0], group, labels, 3)
		self.assertEqual(c, 'B')

	def testKNN2(self):
		dataSet, labels = file2matrix('datingTestSet.txt')
		normDataSet, ranges, minVals = kNN.autoNorm(dataSet)
		testInput = numpy.array([51052, 4.680098, 0.625224])
		testInput = (testInput-minVals) / ranges
		c = kNN.classify0(testInput, normDataSet, labels, 3)
		self.assertEqual(c, 1)

	def testKNN3(self):
		hwLabels = []
		trainingFileList = os.listdir('trainingDigits')
		m = len(trainingFileList)
		trainingMat = numpy.zeros((m,1024))
		for i in range(m):
			fileNameStr = trainingFileList[i]
			fileStr = fileNameStr.split('.')[0]     #take off .txt
			classNumStr = int(fileStr.split('_')[0])
			hwLabels.append(classNumStr)
			trainingMat[i,:] = kNN.img2vector('trainingDigits/%s' % fileNameStr)
		testFileList = os.listdir('testDigits')
		fileNameStr = testFileList[0]
		fileStr = fileNameStr.split('.')[0]     #take off .txt
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = kNN.img2vector('testDigits/%s' % fileNameStr)
		c = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		self.assertEqual(c, 0)


def kNNTestSuite():
	suite = unittest.TestSuite()
	suite.addTest(KNNTestCase("testKNN"))
	suite.addTest(KNNTestCase("testKNN2"))
	suite.addTest(KNNTestCase("testKNN3"))
	return suite

if __name__ == '__main__':
	unittest.main(defaultTest = 'kNNTestSuite')
	#runner = unittest.TextTestRunner()
	#runner.run(suite)

