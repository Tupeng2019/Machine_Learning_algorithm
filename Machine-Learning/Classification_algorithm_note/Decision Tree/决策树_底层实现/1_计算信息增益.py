from math import log

def createDataSet():
    # 创建数据集
    dataSet = [[0, 0, 0, 0, 'no' ],
               [0, 0, 0, 1, 'no' ],
               [0, 0, 1, 1, 'no' ],
               [0, 0, 1, 2, 'yes'],
               [0, 1, 1, 0, 'no' ],
               [0, 1, 1, 1, 'yes'],
               [0, 1, 1, 2, 'yes'],
               [0, 1, 0, 0, 'no' ],
               [0, 1, 1, 2, 'yes'],
               [0, 1, 1, 1, 'yes'],
               [0, 2, 1, 1, 'yes'],
               [0, 2, 1, 1, 'yes'],
               [0, 2, 1, 2, 'yes'],
               [0, 2, 0, 0, 'no' ],
               [0, 2, 1, 2, 'yes'],
               [0, 2, 1, 1, 'yes'],
               [0, 2, 1, 1, 'yes'],
               [1, 1, 1, 1, 'yes'],
               [1, 1, 1, 2, 'yes'],
               [1, 1, 0, 0, 'no' ],
               [1, 1, 1, 2, 'yes'],
               [1, 1, 1, 1, 'yes'],
               [1, 2, 0, 1, 'no' ],
               [2, 0, 0, 1, 'no' ],
               [2, 0, 0, 0, 'no' ],
               [2, 0, 0, 0, 'no' ],
               [2, 0, 1, 2, 'yes'],
               [2, 1, 1, 1, 'yes'],
               [2, 2, 1, 1, 'yes']]
    # 分类属性
    labels = ['age', 'job', 'house','credit']
    # 返回值
    return dataSet,labels

'''
这就是计算经验熵或者叫香农熵
shannon-香农
entropy- 熵
shannonEnt-香农熵
numEntires - 就是返回数据集的行数
calc - 计算
'''
# 编写函数计算信息熵
def calcShannonEnt(dataSet):
    # 返回数据集的总的行数
    numEntires = len(dataSet)
    # 建立一个字典，保存每一个标签label出现的次数
    labelCounts = {}
    # 对每组特征向量进行统计 featVec表示的就是特征向量
    for featVec in dataSet:
        # 提取标签labels的信息，current=现在的
        currentLabel = featVec[-1]
        # 如果标签没有放入统计次数的字典就添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # label 计数
        labelCounts[currentLabel] += 1
    # 经验熵（香农熵）
    shannonEnt = 0.0
    # 计算熵
    for key in labelCounts:
        # 选择该标签label的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用计算熵的公式进行计算
        shannonEnt -= prob * log(prob, 2)
    # 返回熵
    return shannonEnt

# Splitting the dataset 划分数据集
# 函数说明，按照指定的特征划分数据集
# axis= 划分数据集的特征的索引，就相当于下标
# value= 需要返回特征的值

'''
比如我们取solitDataSet(dataSet,0,0)就是将
原数据集中的第一列（索引为0）去掉，然后取剩下的子集为新的数据集（列表）
这是因为我们在计算条件熵的时候，需要知道某个属性的取值，在该样本中，出现的次数
我们取子集的目的就是为了计算某个属性的其中的一个值，发生的次数，作为计算条件熵的参数

'''
def splitDataSet(dataSet, axis, value):
    # 创建返回的数据的列表
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉axis特征，相当于就是取0列到axis-1，列，为新的列表
            reducedFeatVec = featVec[:axis]
            # 将符合条件的添加到返回的数据集中
            # extend()是将括号中的序列直接加到新的列表（的末尾）中间去
            # append（） 是将括号中的整个对象添加到新的列表中，相当于嵌套一个列表
            reducedFeatVec.extend(featVec[axis+1:])
            # 因为retDataSet是一个空列表，所以相当于就是将reducedFeatVec直接放在其中了
            retDataSet.append(reducedFeatVec)
    # 返回划分后的数据集
    return retDataSet


# 选择最优的特征
def chooseBestFeatureToSplit(dataSet):
    # 特征的数量，dataSet[0]表示的返回的是列，所以要减去1
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集的经验熵
    baseEntropy = calcShannonEnt(dataSet)
    # 定义信息增益初始值为0
    bestInfoGain = 0.0
    # 定义最优特征的索引值为-1，就是相当于还没有，应为从0开始的嘛
    bestFeature = -1
    # 遍历所有的特征
    for i in range(numFeatures):
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        # 创建set集合{}，元素是不可以重复的
        uniqueVals = set(featList)
        # 初始化条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集，在者调用上面的划分函数
            subDataSet = splitDataSet(dataSet, i, value)
            #print(subDataSet）
            # 计算子集的概率，这个比较简单
            # 取得就是子集的行数，即为出现的次数
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算条件熵
            # 公式：条件熵=求所有的子集之和（子集的概率*子集的经验熵）
            # 就是i这个属性的value值在总样本中出现的次数
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 求信息增益：总的经验熵- 子集的条件熵
        infoGain = baseEntropy - newEntropy
        # 打印每一个特征的信息增益
        print("第{0}个特征的增益为{1}".format(i, infoGain))

        # 判断信息熵，将最后大的增益选择出来
        if (infoGain > bestInfoGain):
            #做到最大的信息增益
            bestInfoGain = infoGain
            # 同时将信息增益最大的特征的索引记录下来
            bestFeature = i
    # 返回信息增益最大的特征的索引值
    return bestFeature

if __name__ == '__main__':
    dataSet, festures = createDataSet()

    print("最优特征索引值是：" + str(chooseBestFeatureToSplit(dataSet)))
    print('*' *50)

