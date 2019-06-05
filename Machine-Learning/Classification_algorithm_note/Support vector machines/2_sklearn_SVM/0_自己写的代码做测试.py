# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random

"""
数据结构，维护所有需要操作的值
Parameters：
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    kTup - 包含核函数信息的元组,第一个参数存放核函数类别，
           第二个参数存放必要的核函数需要用到的参数
"""


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        # 数据矩阵
        self.X = dataMatIn
        # 数据标签
        self.labelMat = classLabels
        # 定义松弛变量
        self.C = C
        # 容错率
        self.tol = toler
        # 获得矩阵的行数
        self.m = np.shape(dataMatIn)[0]
        # 根据举的的行数初始化alpha参数为0
        self.alphas = np.mat(np.zeros((self.m, 1)))
        # 初始化参数b为0
        self.b = 0
        # 根据矩阵的行数初始化的误差缓存，第一列是否为有效的标志位，
        # 第二列为实际的误差E的值
        self.eCache = np.mat(np.zeros((self.m, 2)))
        # 初始化核参数K
        self.K = np.mat(np.zeros((self.m, self.m)))
        # 计算所有数据的核K
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


"""
通过核函数将数据转换更高维的空间
        X - 数据矩阵
        A - 单个数据的向量
        kTup - 包含核函数信息的元组
        K - 计算的核K
"""


def kernelTrans(X, A, kTup):
    # 获取数据矩阵的行、列数
    m, n = np.shape(X)
    # 初始化第一列，也就是数据的第0列
    K = np.mat(np.zeros((m, 1)))
    # 如果是线性核函数，只进行内积
    if kTup[0] == 'lin':
        K = X * A.T
    # 否者，为高斯核函数，根据高斯核函数公式进行计算
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 计算高斯核K
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('核函数无法识别')
    # 返回计算后得到的核K
    return K


"""
函数说明:读取数据

Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签

 这里的对数据集的处理，和前面都是大同小异，没有什么区别
 可以把前面的直接拿过来用就行了   
"""


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    # 逐行读取数据，
    for line in fr.readlines():
        # 滤除空格等
        lineArr = line.strip().split('\t')
        # 添加数据
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        # 添加标签
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


"""
计算误差

        oS - 数据结构
        k - 标号为k的数据
        Ek - 标号为k的数据误差
"""


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


"""
函数说明:随机选择alpha_j的索引值
    Parameters:
        i - alpha_i的索引值
        m - alpha参数个数
    Returns:
        j - alpha_j的索引值
"""


def selectJrand(i, m):
    # 选择一个不等于i的j
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


"""
内循环启发方式2

    i - 标号为i的数据的索引值
    oS - 数据结构
    Ei - 标号为i的数据误差

    j, maxK - 标号为j或maxK的数据的索引值
    Ej - 标号为j的数据误差
"""


def selectJ(i, oS, Ei):
    # 参数初始化
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    # 根据Ei更新误差缓存
    oS.eCache[i] = [1, Ei]
    # 返回误差不为0的数据的索引值
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    # 当返回的误差值存在是，就是所有不为0的数据
    if (len(validEcacheList)) > 1:
        # 遍历上面返回的数据，从里面找到最大的Ek值
        for k in validEcacheList:
            # 不计算i，浪费时间
            if k == i: continue
            # 计算Ek
            Ek = calcEk(oS, k)
            # 计算|EI=Ek| 的值
            deltaE = abs(Ei - Ek)
            # 判断，找打最大的maxDeltaE
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        # 返回maxK，Ej= Ek
        return maxK, Ej
    # 当前面的情况不存在是，既是没有不为0的误差
    else:
        # 就是随机选择alpha_j的索引值
        j = selectJrand(i, oS.m)
        # 计算Ej
        Ej = calcEk(oS, j)
    return j, Ej


"""
    计算Ek,并更新误差缓存
    Parameters：
        oS - 数据结构
        k - 标号为k的数据的索引值
    Returns:
        无
"""


def updateEk(oS, k):
    # 计算Ek
    Ek = calcEk(oS, k)
    # 更新误差缓存
    oS.eCache[k] = [1, Ek]


'''
clipAlpha()
就是 修剪alpha值
aj = 输入的alpha
H =  alpha的上限
L = alpha的下限
最终的返回值也就是 aj =alpha值

'''


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


"""
优化的SMO算法
        i - 标号为i的数据的索引值
        oS - 数据结构
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小
"""


def innerL(i, oS):
    # 步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    # 优化alpha,设定一定的容错率。,这个就是和前面的公式是一样的了
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2 选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        # 深拷贝就是相当于创建一个新的列表和原来的是一模一样的，
        # 当原来的改变后，深拷贝的是不会改变的
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        # 步骤2：计算上下界H和L
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)
        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


"""
完整的线性SMO算法
        dataMatIn - 输入的数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
        maxIter - 最大迭代次数
        kTup - 包含核函数信息的元组

        oS.b - SMO算法计算的b
        oS.alphas - SMO算法计算的alphas
"""


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    # 初始化数据结构
    # 这就是前面定义的optStruct的类，这里就是初始化数据
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    # 初始化迭代次数
    iter = 0
    # entireSet表示的就是全部的数据集
    # alphaPairsChangeed 表示的就是alpha优化的次数
    entireSet = True;
    alphaPairsChanged = 0
    # 判断，遍历整个数据集的alpha也没有更新或者最大迭代次数，则退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # 遍历整个数据集
        if entireSet:
            for i in range(oS.m):
                # 使用刚刚前面的优化SMO算法
                alphaPairsChanged += innerL(i, oS)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 遍历非边界值
        else:
            # 遍历不在边界0和C的alpha
            '''
            nonzero(a)返回数组a中值不为零的元素的下标，
            它的返回值是一个长度为a.ndim(数组a的轴数)的元组，
            元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值
            当时二维数组的时候，就应该是以数组的形式来看，对应的看结果
            '''
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 遍历一次后改为非边界值
        if entireSet:
            entireSet = False
        # 如果alpha没有更新，计算全样本遍历
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("迭代次数: %d" % iter)
    return oS.b, oS.alphas

"""
将32x32的二进制图像转换为1x1024向量。
    Parameters:
        filename - 文件名
    Returns:
        returnVect - 返回的二进制图像的1x1024向量
"""
def img2vector(filename):
    # 初始化返回向量
    returnVect = np.zeros((1,1024))
    # 读取数据集文件
    fr = open(filename)
    # 因为我们的数据集是32 * 32 的
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


"""
该函数：就是加载图片的作用
    Parameters:
        dirName - 文件夹的名字
    Returns:
        trainingMat - 数据矩阵
        hwLabels - 数据标签
"""
def loadImages(dirName):

    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


"""
    测试函数
    Parameters:
        kTup - 包含核函数信息的元组
    Returns:
        无
"""

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print("支持向量个数:%d" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("训练集错误率: %.2f%%" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("测试集错误率: %.2f%%" % (float(errorCount)/m))

if __name__ == '__main__':
    testDigits()
