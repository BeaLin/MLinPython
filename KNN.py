#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from numpy import *
import operator

'''
inX:用于分类的输入向量是inX
dataSet:训练样本集
labels:标签向量
k:用于选择最近邻居的数目
'''
def classify(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    #tile(A,(n,m)):将A重复n*m次变成n*m二维数组
    #使用欧式距离公式计算两个向量点之间的距离
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #对距离数组进行排序，argsort()返回数组值从小到大的索引值
    sortedDistIndicies=distances.argsort()
    classCount={}
    #取前k个
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel, 0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

group,labels=createDataSet()
print classify([0,0],group,labels,3)



