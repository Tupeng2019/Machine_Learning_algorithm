import numpy as np
import operator
'''
# 步骤
    定义一个分类器
        这里运用k近邻的算法公式进行编程
        将数据分为测试集和训练集
        对训练集进行训练
        最后岁测试集进行测试
        返回测试集的最终的测试结果
'''
# 建立函数，创建数据集，这只是最初的一小步
def createDataSet():
    # 建立几个不同的二维数组，可以是随意的，group
    group = np.array([[1,101],[8,158],[245,9],[112,25]])
    # 对以上的四个二维数组进行标签处理label
    labels = ["不好吃","不好吃","好吃" ,"好吃"]
    # 函数一般都会有返回值
    return group, labels

# 构建分类器classify0
# inX 表示的是用于分类的测试集
# data表示的就是训练数据集
#labels 就是上面一样的标签
# k 表示利用算法所得到的距离最小的k个点
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

# 定义主函数
if __name__ == '__main__':
    # 进行数据集的实例化
    group,labels =createDataSet()
    # 输入测试集
    test = [145,9]
    # 进行KNN实例化
    Test_class = classify0(test,group,labels,4)
    # 打印最终的分类结果
    print(Test_class)

