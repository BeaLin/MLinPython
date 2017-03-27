#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from numpy import *

#Sigmoid函数，
# 当x为0时，Sigmoid函数值为0.5，
# 随着x的增大，对应的Sigmoid值将逼近1，
# 而随着x的减小，Sigmoid值将逼近于0.
# 当横坐标刻度足够大，Sigmoid函数像一个阶跃函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法函数
#dataMatIn:2维NumPy数组，每列分别代表每个不同的特征，每行代表每个训练样本。
#classLabels：类别标签，是一个1*100的行向量
def gradAscent(dataMatIn,classLabels):
    #变为矩阵
    dataMatrix=mat(dataMatIn)
    #为了便于矩阵运算，需要将该行向量转置为列向量
    labelMat=mat(classLabels).transpose()
    #返回矩阵的行数、列数
    m,n=shape(dataMatrix)
    #步长为0.001
    alpha=0.001
    #迭代次数为500
    maxCycles=500
    #初始化回归系数
    weights=ones((n,1))
    for k in range(maxCycles):
        #在每个特征上乘以一个回归系数，然后把所有的结果值相加，
        # 将总和代入Sigmoid函数，进而得到一个在0~1之间的数值，大于0.5被分入1类，小于0.5被归入0类
        h=sigmoid(dataMatrix*weights)
        #计算真实类别与预测类别的差值
        error=(labelMat-h)
        #按照该差值的方向调整回归系数
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

#随机梯度上升算法
#梯度上升算法每次更新回归系数都需要遍历整个数据集，若数据量大或者特征多的时候，计算复杂度就太高了
#该改进方法是一次仅用一个样本点来更新回归系数
#由于可以在新样本到来时对分类器进行增量式更新，故随机梯度上升算法是一个在线学习算法
def stocGradAscent0(datamatrix,classLabels):
    m,n=shape(datamatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(datamatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*datamatrix[i]
    return weights

#改进的随机梯度上升算法
#改进1：增加alpha，其在每次迭代时都会调整，可缓解回归系数的波动或者高频波动
#改进2：通过随机选取样本来更新回归系数，可减少周期性的波动
#numIter:默认参数，迭代次数默认为150
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            #参数alpha会随着迭代次数不断减小，但不会减小到0，保证在多次迭代之后新数据仍然具有一定的影响
            #若要处理的问题是动态变化的，那么可适当增大常数项，即0.01
            #当j<<max(i)时，alpha不是严格下降，避免参数的严格下降也常用于模拟退火算法等其他优化算法中
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
#用Logistic回归进行分类
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

#测试梯度上升算法，求回归系数
#载入数据
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('data/testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#画出决策边界
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
#疝气病症预测病马的死亡率
def colicTest():
    frTrain=open('data/horseColicTraining.txt')
    frTest=open('data/horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!= int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount/numTestVec))
    print "The error rate of this test is: %f" % errorRate
    return errorRate
def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests))
#测试梯度上升算法
#dataArray,labelMatrix=loadDataSet()
#weights=gradAscent(dataArray,labelMatrix)
#plotBestFit(weights.getA())
#测试随机梯度上升算法
#weights=stocGradAscent0(array(dataArray),labelMatrix)
#plotBestFit(weights)
#测试改进的随机梯度上升算法
#weights=stocGradAscent1(array(dataArray),labelMatrix)
#plotBestFit(weights)
#预测病马的死亡
multiTest()





