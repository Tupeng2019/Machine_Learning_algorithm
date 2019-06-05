from  math import log
import matplotlib.pyplot as plt
import operator

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

'''
建立函数：统计类标签classList中出现此处最多的元素
majority 多数的
majorityCnt就是定义函数表示出现次数的计算

'''
def majorityCnt(classList):
    # 初始化一个新的字典
    classCount = {}
    # 统计classList中每一个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 根据字典的值降序排序
    # .items()表示就是classList中所有的元素
    # key=.. 表示的就是定义一个函数key，获取对象的第一个域中的值，就是相当于取classList中第一个值
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    # 返回classList中出现次数最多的元素
    # 因为返回的sortedClassCount 是一个list，但是其中的元素是就是tuple类型，所以用[0][0]
    return sortedClassCount[0][0]

'''
创建决策树
createTree：建立决策树函数
festLabels: 就是定义的最优的特征标签

'''
def createTree(dataSet, labels, featLabels):
    # 取分类标签（是否买车yes or no）
    # 将dataSet中的数据集按行的格式放入example中，然后取example中example[-1]的元素
    # 而这里的example[i]就是第i-1列的元素，而不是取行了
    # 放入classLIst中去，-1 就是表示原数据的最后一列，即为no，或者yes
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同则停止继续划分
    # 这里的类标签指的就是最后的yes ，no
    # count 就是指出现这个类标签的次数等于总的类标签的个数
    # 这两个if 就是我们tree结束的最后条件
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征就返回次数最多的类标签
    # 这就是表示所有的特征白哦前都已经划分完了，只有yes，no 这个标签了
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最优特征的标签
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    # 根据最优特征的标签生成树
    myTree = {bestFeatLabel:{}}
    #删除已经选择过得特征标签
    del(labels[bestFeat])
    # 得到训练集中所有的最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
    # 去掉重复的属性值，set就是构建一个无序的独一无二的集合，就是去重吧
    uniqueVals = set(featValues)
    # 遍历特征，创建决策树
    # 相当于就是一个循环，一直到划分完事
    for value in uniqueVals:
        # mytree 就是一个字典类型，其元素就是tuple类型的
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    # 返回自己建立的决策树
    return myTree



'''
上面的代码，都是在构建决策树
而现在以后的代码就是在使用决策树进行分类了
classify 就是一个分类的函数

testVec 表示的就是进行测试的数据集
'''

def classify(inputTree, featLabels, testVec):
    # 获取决策树的结点,是以字典的形式保存在firstStr中，
    # next(iterator[, default])
    # Return the next item from the iterator. If default is given and the iterator
    # is exhausted, it is returned instead of raising StopIteration.
    # 译：=》 返回迭代器的下一项，如果给出默认值，如果迭代器已经用尽，则返回它，
    # 而不是"停止迭代"
    # 这里的firstStr就是job，相当于就是（键值对中的键）
    firstStr = next(iter(inputTree))
    # 转到下一个字典
    # 这里的secondDict就是job下面的整个字典
    # {0: 'no', 1: {'credit': {0: 'no', 1: {'house': {0: 'no', 1: 'yes', 2: 'yes'}}, 2: 'yes'}}}
    secondDict = inputTree[firstStr]
    # 这里就是返回的最优特征的标签(就是按照树的形状的从上倒下来取得）
    # ['job', 'credit', 'house']
    featIndex = featLabels.index(firstStr)
    # secondDict.key就是返回的是job下的键值[0,1]
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                # 这一步就是一个梯归的操作，迭代到credit这个特征标签进行操作。
                classLabel = classify(secondDict[key], featLabels, testVec)
            # 就是将最终的分类yes or no 依次赋值给classLabel
            else: classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet,labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels,featLabels)
    print(myTree)
    # 在这里的输入就是最优特征的一次输入，
    # 是按照最优的特征标签一次输入的，可以只输入三个数据，就好了
    testVec = [ 1,1,2 ]
    result = classify(myTree, featLabels,testVec)
    if result == 'yes':
        print('*'* 50)
        print("the people can buy a BMW")
    if result == 'no':
        print('*'* 50)
        print("the people can't buy a BMW")

