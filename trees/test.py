#!/usr/bin/env python
#encoding=utf-8

import unittest
import trees

class TreesTestCase(unittest.TestCase):

	@staticmethod
	def createDataSet():
		dataSet = [[1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']]
		labels = ['no surfacing','flippers']
		#change to discrete values
		return dataSet, labels

	def setUp(self):
		pass

	def testCalcShannonEnt(self):
		myDat,labels = TreesTestCase.createDataSet()
		entropy = trees.calcShannonEnt(myDat)
		ent = float('%0.3f' % entropy)
		self.assertEqual(ent, 0.971)

	def testSplitDataSet(self):
		myDat,labels = TreesTestCase.createDataSet()
		part1 = trees.splitDataSet(myDat, 0, 1)
		part2 = trees.splitDataSet(myDat, 0, 0)
		self.assertEqual(part1, [[1, 'yes'], [1, 'yes'], [0, 'no']])
		self.assertEqual(part2, [[1, 'no'], [1, 'no'],])

	def testChooseBestFeatureToSplit(self):
		myDat,labels = TreesTestCase.createDataSet()
		featureIndex = trees.chooseBestFeatureToSplit(myDat)
		self.assertEqual(featureIndex, 0)

	def testCreateTree(self):
		myDat,labels = TreesTestCase.createDataSet()
		tree = trees.createTree(myDat, labels)
		theTree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
		self.assertEqual(theTree, tree)

	def testClassify(self):
		myDat,labels = TreesTestCase.createDataSet()
		tree = trees.createTree(myDat, labels)
		c = trees.classify(tree, labels, [1, 0])
		self.assertEqual(c, 'no')
		c = trees.classify(tree, labels, [1, 1])
		self.assertEqual(c, 'yes')


#not use
def treesTestSuite():
	suite = unittest.TestSuite()
	suite.addTest(TreesTestCase("testCalcShannonEnt"))
	suite.addTest(TreesTestCase("testSplitDataSet"))
	suite.addTest(TreesTestCase("testChooseBestFeatureToSplit"))
	suite.addTest(TreesTestCase("testCreateTree"))
	suite.addTest(TreesTestCase("testClassify"))
	return suite

if __name__ == '__main__':
	# 执行treesTestSuite中的测试用例
	#unittest.main(defaultTest = 'treesTestSuite')

	#会自动查找当前文件中所有的测试用例并执行
	unittest.main()
	#runner = unittest.TextTestRunner()
	#runner.run(suite)

