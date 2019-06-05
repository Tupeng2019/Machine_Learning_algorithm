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

'''
mutIt is the number of iteration：迭代器的数量

'''
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)   #初始化权重
    aggClassEst = np.mat(np.zeros((m,1)))
    # this is heart of AdaBoost algorithm
    for i in range(numIt):
        # 构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:",D.T)
        #计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        # 存储弱学习算法权重
        bestStump['alpha'] = alpha
        # 存储单层决策树
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        '''
        这下面的三行就是用来计算下一次迭代的权重值
        '''
        # 计算的e的指数项
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        # 根据样本得到权重公式，更新样本的权重
        D = D / D.sum()
        #计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        # 计算误差
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        # 判断误差，误差是0时，退出循环
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

"""
AdaBoost分类函数
    Parameters参数:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
    Returns返回结果:
        分类结果
"""

def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    # 记录每个数据点的类别估计累计值。
    aggClassEst = np.mat(np.zeros((m,1)))
    # 遍历多有的分类器，进行分类
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)
if __name__ == '__main__':
    dataArr,classLabels = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    print("*" * 80)
    print(adaClassify([[5,5],[0,0]], weakClassArr))
