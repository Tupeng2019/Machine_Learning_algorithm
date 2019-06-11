# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

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
该函数：绘制ROC
    Parameters:
        predStrengths - 分类器的预测强度
        classLabels - 类别
    Returns:
        无
"""
def plotROC(predStrengths, classLabels):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 绘制光标的位置
    cur = (1.0, 1.0)
    # 这就是用于计算AUC，既是曲线下面的面积
    ySum = 0.0
    # 统计正类的数量
    numPosClas = np.sum(np.array(classLabels) == 1.0)
    # 计算y轴的步长
    yStep = 1 / float(numPosClas)
    # 计算x轴的步长
    xStep = 1 / float(len(classLabels) - numPosClas)
    # 对预测强度的排序
    sortedIndicies = predStrengths.argsort()
    # 绘图
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            # 高度的累加y
            ySum += cur[1]
        # 绘制ROC
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
        # 更新绘制光标的位置
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0,1], [0,1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties = font)
    plt.xlabel('假阳率', FontProperties = font)
    plt.ylabel('真阳率', FontProperties = font)
    ax.axis([0, 1, 0, 1])
    # 计算并打印AUC
    print('AUC面积为:', ySum * xStep)
    plt.show()


if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr, 10)
    plotROC(aggClassEst.T, LabelArr)

'''
我们可以看到有两个输出结果，一个是AUC面积，另一个ROC曲线图。
-  先解释ROC，图中的横坐标是伪正例的比例（假阳率=FP/（FP+TN）），而纵坐标是真正例的比例（
   真阳率=TP/（TP+FN））。ROC曲线给出的是当阈值变化时假阳率和真阳率的变化情况。左下角的点
   所对应的将所有样例判为反例的情况，而右上角的点对应的则是将所有样例判为正例的情况。
   虚线给出的是随机猜测的结果曲线。
   
-  ROC曲线不但可以用于比较分类器，还可以基于成本效益（cost-versus-benefit）分析来做出决策。
   由于在不同的阈值下，不用的分类器的表现情况是可能各不相同，因此以某种方式将它们组合起来
   或许更有意义。如果只是简单地观察分类器的错误率，那么我们就难以得到这种更深入的洞察效果了
   
-  在理想的情况下，最佳的分类器应该尽可能地处于左上角，这就意味着分类器在假阳率很低的同时
   获得了很高的真阳率。例如在垃圾邮件的过滤中，就相当于过滤了所有的垃圾邮件，但没有将任何
   合法邮件误识别为垃圾邮件而放入垃圾邮件额文件夹中。
   
-  对不同的ROC曲线进行比较的一个指标是曲线下的面积（Area Unser the Curve，AUC）。AUC给出的
   是分类器的平均性能值，当然它并不能完全代替对整条曲线的观察。一个完美分类器的ACU为1.0，
   而随机猜测的AUC则为0.5。
'''