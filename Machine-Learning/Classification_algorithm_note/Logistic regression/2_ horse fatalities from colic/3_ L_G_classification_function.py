# -*- coding:UTF-8 -*-

import numpy as np
import random

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

'''
下面就是我们的梯度上升算法的实现
gradAscent
'''
def gradAscent(dataMatIn, classLabels):
    # 将数据集转换为numpy的mat
    dataMatrix = np.mat(dataMatIn)
    # 将标签转换成numpy的mat。并且进行转置
    # transpose 就是表示转置
    labelMat = np.mat(classLabels).transpose()
    # 返回dataMatrix的大小。m为行数,n为列数。
    m, n = np.shape(dataMatrix)
    # 移动步长,也就是学习速率,控制更新的幅度。
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # 权重的初始化为0
    weights = np.ones((n,1))
    for k in range(maxCycles):
        # h就是代表z的含义，梯度上升矢量化公式
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转换为数组，返回权重数组
    # 这里的最优权重数组就是所求的最佳的解
    return weights.getA()


'''
下面就是我们的随机梯度上升算法梯度上升算法的实现
stocGradAscent
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 返回dataMatrix的大小。m为行数,n为列数。
    m,n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 降低alpha的大小，每次减小1/(j+i)。
            # 相比如前面所做的改变：Alpha在每一次迭代中，都进行了改变
            alpha = 4/(1.0+j+i)+0.01
            # 随机选取样本
            # 这里也是做出的改变：是随机的选择样本，而不是遍历所有的样本
            randIndex = int(random.uniform(0,len(dataIndex)))
            # 随机选取的一个样本，计算h
            # h就是代表z的含义，梯度上升矢量化公式
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 将已经使用过得样本进行删除
            del(dataIndex[randIndex])
    # 返回权重
    return weights

'''
这就是：分类函数
inX - 特征向量
weights - 回归系数（就是w，相当于权重）
'''

def classifyVector(inX, weights):
    # 计算Sigmoid函数的值
    # 当值是>0.5时，就是分为1类
    # 当值是<0.5时，就是分为0类
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


'''
这就是我们的colic的测试函数
'''
def colicTest():
    # 这就是读取训练集
    frTrain = open('horseColicTraining.txt')
    # 读取测试集
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    # 使用改进后的随机梯度上升算法
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)

    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
            errorCount += 1
    # 计算错误率
    errorRate = (float(errorCount)/numTestVec) * 100
    print("测试集错误率为: %.2f%%" % errorRate)

if __name__ == '__main__':
    colicTest()