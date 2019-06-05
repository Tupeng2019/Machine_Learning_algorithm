# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt



def loadSimpData():

    datMat = np.matrix([[1., 2.1],
                        [1.5, 1.6],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

"""
单层决策树分类函数
        dataMatrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        threshVal - 阈值
        threshIneq - 标志
    返回值：
        retArray - 分类结果
"""

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    # 初始化retArray为1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    # 这就是设置阈值
    if threshIneq == 'lt':
        # 如果小于阈值，则赋值为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        # 如果大于阈值，则赋值为-1
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

"""
找到数据集上最佳的单层决策树
     Parameters参数:
         dataArr - 数据矩阵
         classLabels - 数据标签
         D - 样本权重
     Returns返回值:
         bestStump - 最佳单层决策树信息
         minError - 最小误差
         bestClasEst - 最佳的分类结果
"""

def buildStump(dataArr, classLabels, D):

    dataMatrix = np.mat(dataArr);
    # 取矩阵转置
    labelMat = np.mat(classLabels).T
    # 读取矩阵的行列
    m, n = np.shape(dataMatrix)
    numSteps = 10.0;bestStump = {};bestClasEst = np.mat(np.zeros((m, 1)))
    # 将最小误差是初始化为正无穷大
    minError = float('inf')
    # 遍历所有的特征，也就是数据集中的所有列
    for i in range(n):
        # 找到特征中最小的值和最大值
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()
        # 计算步长
        stepSize = (rangeMax - rangeMin) / numSteps  #
        for j in range(-1, int(numSteps) + 1):
            # 大于和小于的情况，均遍历。lt:less than 小于，gt:greater than大于
            for inequal in ['lt', 'gt']:
                # 计算阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 记录分类的结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 将误差矩阵去啊女不初始化为0
                errArr = np.mat(np.ones((m, 1)))
                # 分类正确的，赋值为0
                errArr[predictedVals == labelMat] = 0
                # 计算误差
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                i, threshVal, inequal, weightedError))
                # 找到最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


if __name__ == '__main__':
    dataArr, classLabels = loadSimpData()
    D = np.mat(np.ones((5, 1)) / 5)
    bestStump, minError, bestClasEst = buildStump(dataArr, classLabels, D)
    print('bestStump:\n', bestStump)
    print('minError:\n', minError)
    print('bestClasEst:\n', bestClasEst)
