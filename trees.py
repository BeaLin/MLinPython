#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from math import log
import operator
'''
**构建决策树进行分类
该决策树代码对输入数据的要求是：
1.数据必须是一种由列表元素组成的列表，而且所有的列表袁术都要具有相同的数据长度
2.数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签
'''
'''
计算给定数据集的香农熵
'''
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    #构建数据字典，记录数据集中出现的类别及其出现次数
    labelCounts={}
    #判断数据集中出现的标签是否已经存在在数据字典中。存在则value值加1，不存在则扩展字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    #根据工商计算香农熵
    for key in labelCounts:
        prob =float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
'''按照给定特征划分数据集
dataSet:待划分的数据集
axis:划分数据集的特征
value:需要返回的特征的值
返回的数据集中不再包含该划分数据集的特征的值
'''
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
'''选择最好的数据集划分方式'''
def chooseBestFeatureToSplit(dataSet):
    #获取数据集的特种属性的数量
    numFeatures = len(dataSet[0])-1
    #计算整个数据集的原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    #遍历所有的特征，并计算根据每一个特征划分数据集后的新香农熵
    for i in range(numFeatures):
        #获取第i个特征的所有值
        featList=[example[i] for example in dataSet]
        #使用集合对列表中的值进行去重
        uniqueVals=set(featList)
        newEntropy=0.0
        #这个for循环计算根据第i个特征划分的新数据集的香农熵
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        #获得最大的信息增益下的i
        infoGain =baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
'''
返回出现次数最多的分类名称
classList:分类名称列表
'''
def majorityCnt(classList):
    #字典对象村粗classList中每个类标签出现的频率
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
        sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]
'''递归构建决策树'''
def createTree(dataSet,labels):
    #获取类标签
    classList=[example[-1] for example in dataSet]
    #递归函数的停止条件1：所有的类标签完全相同，即完全分类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #递归函数的停止条件2：使用完所有特征，仍不能将数据集划分成仅包含唯一类别的分组
    #无法简单返回唯一的类标签，故返回出现 次数最多的类标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #获得最好的特征索引
    bestFeat=chooseBestFeatureToSplit(dataSet)
    #获得最好的特征标签
    bestFeatLabel=labels[bestFeat]
    #构建决策树
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    #递归调用创建决策树
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return  myTree
#创建数据集
def creatDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels
'''使用决策树的分类函数'''
def classify(inputTree,featLabels,testVec):
    firstStr=inputTree.keys()[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

#测试计算香农熵
'''myDat,labels=creatDataSet()
print myDat
print calcShannonEnt(myDat)
myDat[0][-1]='maybe'
print myDat
print calcShannonEnt(myDat)'''
#测试划分数据集
'''myDat,labels=creatDataSet()
print myDat
print splitDataSet(myDat,0,1)
print splitDataSet(myDat,0,0)
print splitDataSet(myDat,1,0)'''
#测试数据集划分的选择
'''myDat,labels=creatDataSet()
print myDat
print chooseBestFeatureToSplit(myDat)'''
#测试创建决策树
'''myDat,labels=creatDataSet()
print myDat
print createTree(myDat,labels)'''
#测试分类
myDat,labels=creatDataSet()
myDat,myLabels=creatDataSet()
myTree=createTree(myDat,labels)
print classify(myTree,myLabels,[1,0])
print classify(myTree,myLabels,[1,1])
print classify(myTree,myLabels,[0,1])
print classify(myTree,myLabels,[0,0])