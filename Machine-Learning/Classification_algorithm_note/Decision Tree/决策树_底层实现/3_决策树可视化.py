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
获取决策树叶子节点的数目
numLeafs = 决策树叶子节点的个数

'''


def getNumLeafs(myTree):
    # 初始化叶子节点的数目
    numLeafs = 0  # 初始化叶子
    # python3中myTree.keys()返回的是dict_keys,不在是list,
    # 所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    # 获取下一组字典
    secondDict = myTree[firstStr]
    # 第三阻字典
    #thirdDict = myTree[secondDict]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
                numLeafs += getNumLeafs(secondDict[key])
        else:
                numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    # 初始化决策树的深度
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    # 一下的做法就是判断，如果是一个dict，这就是说明是一个叶节点
    for key in secondDict.keys():
        # 判断是否是dict类型
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        # 更新层数
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

# 绘制结点位置
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 指定箭头的格式
    arrow_args = dict(arrowstyle="<-")
    # 设置中文字体
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 绘制结点， annontate =注释
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, )


# 标注有向边属性值
'''
《Machine_Learning in action》
The code in plotMidText() calculates the midpoint between 
the parent and child nodes and puts a simple text label in the middle
这个plotMIdText()就是计算父节点与子节点的中间位置，
并且在中间位置放置一个简单的文本标签
'''
def plotMidText(cntrPt, parentPt, txtString):
    # 计算标记的位置
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
    # 设置结点格式
    # decisionNode就是决策结点，就是相当于每一个父结点，sawtooth=锯齿状。
    # 这里的fc就是表示其灰度值，当为0是就是黑色，1是就是白色
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    # leafNode叶节点，就是yes。no，round便是的是圆形
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树的叶节点数目，这将决定书的宽度
    numLeafs = getNumLeafs(myTree)
    # depth就是表达深度，也即是树的层数
    depth = getTreeDepth(myTree)
    # 转到下一个字典，
    firstStr = next(iter(myTree))
    # 计算中心位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 调用plotNode函数，绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 转到下一个字典，继续绘制子结点，这里使用的梯归调用
    secondDict = myTree[firstStr]
    # 计算y偏移
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        # 测试该节点是否为字典，入股不是就是叶子节点，如果是则继续绘制节点
        if type(secondDict[key]).__name__=='dict':
            # 如果是字典，就使用梯归调用继续向下操作
            plotTree(secondDict[key],cntrPt,str(key))
        # 如果是叶子节点就直接绘制叶子结点，并标注指向属性值
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            # 直接调用前面定义的函数，既是重复的操作，只是到下一层而已
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


# 绘制绘图面板，
# 这里的inTree就是一个形参，是不用太多考虑的，后面的实参就是myTree
def createPlot(inTree):
    # fig就是指创建一个图形的窗口，facecolor表示的就是背景颜色
    fig = plt.figure(1, facecolor='white')
    # Clear the figure 就是将窗口清空
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 将窗口的原来的坐标轴全部去掉
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # totalW就是表示宽度，就是获取决策树的叶节点的数目，叶节点决定书的宽度
    plotTree.totalW = float(getNumLeafs(inTree))
    # totalD就是表示深度，就是获取树的层数
    plotTree.totalD = float(getTreeDepth(inTree))
    # x的偏移
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    # 绘制树
    plotTree(inTree, (0.5,1.0), '')
    # 显示图像
    plt.show()



if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
    createPlot(myTree)
