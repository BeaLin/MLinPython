#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from numpy import *
'''创建实验样本
返回的第一个变量是进行词条切分后的文档集合
返回的第二个变量是类别标签的集合'''
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','streak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1] #1代表侮辱性文字，0代表正常言论
    return postingList,classVec
'''创建一个包含在所有文档中出现的不重复词的列表'''
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)
'''
朴素贝叶斯词集模型：将某个词的出现与否作为一个特征
输入参数为词汇表及某个文档，输出的是文档向量
向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现。'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec
'''朴素贝叶斯词袋模型：在词袋中，每个词可以出现多次，在词集中，每个词集只能出现一次'''
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec
'''朴素贝叶斯分类器训练函数
输入文档矩阵和每篇文档类别标签所构成的向量'''
def trainNBO(trainMatrix,trainCategory):
    #训练文档的个数
    numTrainDocs=len(trainMatrix)
    #词汇表的个数
    numWords=len(trainMatrix[0])
    #文档中属于侮辱类的概率为0.5
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #初始化为0
    '''p0Num=zeros(numWords)
    p1Num=zeros(numWords)
    p0Denom=0.0
    p1Denom=0.0'''
    '''根据现实情况修改分类器：利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积，
    如果其中一个概率值为0，则最后的结果会受影响也为0，
    故将所有词的出现次数初始化为1，分母初始化为2'''
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    #遍历训练集trainMatrix中的所有文档，如果这个文档为侮辱类别，则该词为侮辱性的个数加1，即p1Num加1，且总词数也加1，否则为正常的加
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    '''p1Vect=p1Num/p1Denom
    p0Vect=p0Num/p0Denom'''
    '''太多很小的数相乘会导致下溢出，而导致得到不正确的答案
    可以通过球对数避免下溢出或浮点数四射五入导致的错误
    采用自然对数进行处理不会有损失，可以通过观察f(x）和ln(f(x))的图像'''
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNBO(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)
#测试创建的实验样本
'''listOPosts,listClasses=loadDataSet()
#print  listOPosts
myVocabList=createVocabList(listOPosts)
#print  myVocabList
#print setOfWords2Vec(myVocabList,listOPosts[0])
#print setOfWords2Vec(myVocabList,listOPosts[3])
trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
#print trainMat
#测试贝叶斯分类器训练函数
p0V,p1V,pAb=trainNBO(trainMat,listClasses)
print p0V
print p1V
print pAb'''
#测试朴素贝叶斯分类
testingNB()