#!/usr/bin/env python
#encoding=utf8

import unittest
import bayes

class BayesTestCase(unittest.TestCase):

	@staticmethod
	def loadDataSet():
		postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
		classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
		return postingList,classVec

	def setUp(self):
		pass

	def testSetOfWords2Vec(self):
		trainSet, classVec = BayesTestCase.loadDataSet()
		vocabList = bayes.createVocabList(trainSet)
		vec = bayes.setOfWords2Vec(vocabList, trainSet[0])
		theVec = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]
		self.assertEqual(theVec, vec)

if __name__ == '__main__':
	unittest.main()

