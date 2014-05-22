#!/usr/bin/env python
# encoding=utf8

def p(name, obj):
    print name + ':'
    print repr(obj)

from numpy import *

dataMatrix = mat ([[1,2,3,4],[4,5,6,7],[8,9,10,11]])
labelMat = mat([[1,2,3]])
p('dataMatrix', dataMatrix)
p('labelMat', labelMat)

b = 0
m, n = shape(dataMatrix)
alphas = mat(zeros((m, 1)))
p('alphas', alphas)

p('alphas * labelMat',alphas * labelMat )

i = 0
p('dataMatrix*dataMatrix[i,:].T', dataMatrix*dataMatrix[i,:].T)
