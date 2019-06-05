from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np



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

if __name__ == '__main__':
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    normDataSet, ranges, minValues = autoNorm(datingDataMat)
    print(normDataSet)
    print('*' *20)
    print(ranges)
    print('*'*20)
    print(minValues)
