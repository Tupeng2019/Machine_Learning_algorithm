from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator



def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # 表示返回训练集的行数
    # 就是相当于做距离的差在平方，最后相加
    # diffMat 就是一个扩散器
    # np.tile 就是将inX扩展成一个四行两列的数组然后分别于dateSet做距离减法
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 就是将距离差，做平方和
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)表示相应的列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]



#  这就进行数据的解析，这一般就是我们进行KNN，或者别的机器学习的第一步
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    arrayOLines = fr.readlines()
    #得到文件行数
    numberOfLines = len(arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    # 这是因为我们的女主只有三个指标
    returnMat = np.zeros((numberOfLines,3))
    #返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike': # 不喜欢
            classLabelVector.append("didntLike")
        elif listFromLine[-1] == 'smallDoses':  # 有一点魅力，魅力一般
            classLabelVector.append("smallDoses")
        elif listFromLine[-1] == 'largeDoses':  # 非常有魅力
            classLabelVector.append("largeDoses")
        index += 1
    return returnMat, classLabelVector


'''
下面就是对数据进行归一化的操作
利用公式： newValue = (oldValue - min)/ (max - min)

'''
def autoNorm(dataSet):    # dataSet 就是特征矩阵
    #获得数据最小值
    minValues = dataSet.min(0)
    # 获得数据最大值
    maxValues = dataSet.max(0)
    # 最大值和最小值的范围
    ranges = maxValues - minValues
    # shape(dataSet)返回dataSet的矩阵行列数
    # normDataSet 就是归一化后的矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回dataSet的行数
    m = dataSet.shape[0]
    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minValues, (m, 1))
    # 除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minValues




# 分类器测试函数，用于对数据即的测试
def datingClassTest():
    # 取所有数据的百分之十
    hoRatio = 0.10
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minValues = autoNorm(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0.0
    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        # 中间的星号就是便于观察没哟什么实际的意义
        print("the classifier came back with:{0}, **  the real answer is:{1}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is:%f" % (errorCount / float(numTestVecs) * 100))

if __name__ =='__main__':
    #打开的文件名
    filename = "datingTestSet.txt"
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    datingClassTest()




