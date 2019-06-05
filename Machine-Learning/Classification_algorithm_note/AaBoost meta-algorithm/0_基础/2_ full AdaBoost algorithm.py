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
    D = np.mat(np.ones((m, 1)) / m)                     #初始化权重
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

if __name__ == '__main__':
    dataArr,classLabels = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    print("*" * 50)
    print(weakClassArr)
    print("*" * 50)
    print(aggClassEst)
'''
********  这里就是对运行结果的解释：

在第一轮迭代中，D中的所有值都相等。于是，只有第一个数据点被错分了。
因此在第二轮迭代中，D向量给第一个数据点0.5的权重。这就可以通过变量aggClassEst的符号
来了解总的类别。第二次迭代之后，我们就会发现第一个数据点已经正确分类了，
但此时最后一个数据点却是错分了。D向量中的最后一个元素变为0.5，而D向量中的其他值都变得
非常小。最后，第三次迭代之后aggClassEst所有值的符号和真是类别标签都完全吻合，
那么训练错误率为0，程序终止运行。

最后训练结果包含了三个弱分类器，其中包含了分类所需要的所有信息。一共迭代了3次，
所以训练了3个弱分类器构成一个使用AdaBoost算法优化过的分类器，分类器的错误率为0

'''