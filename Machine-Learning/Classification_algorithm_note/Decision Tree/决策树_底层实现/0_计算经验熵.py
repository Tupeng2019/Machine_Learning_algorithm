from math import log


'''
定义函数
创建数据集
编码计算信息熵
 # 这里举的例子就是
   “判断一个人是否能买得起宝马车，从，年龄、工作、是否有房子，信贷情况”
   年龄：0 就是18-30， 1 就是30-45岁，2代表就是45-65
   工作：0代表没工作，1代表有工作，2代表工作非常好，
   房子：0 代表没有房子，1 代表有房子
   信贷情况：0 代表不好，1代表一般，2代表很好
'''
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

if __name__== '__main__':
    dataSet, features = createDataSet()
    print('*' *30)
    print(dataSet)
    print('*'* 30)
    print(calcShannonEnt(dataSet))

