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



# 定义显示图像函数
def showDatas(datingDataMat, datingDataLabels):

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 'didntLike':
            LabelsColors.append('blue')
        if i == 'smallDoses':
            LabelsColors.append('black')
        if i == 'largeDoses':
            LabelsColors.append('red')
    # 就是生成图像
    image1 = plt.subplot()
    # 画出散点图,以datingDataMat矩阵的第一列(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    image1.scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1],color=LabelsColors ,s=10, alpha=1)
    # 设置标题,x轴label,y轴label
    image1_title_text = image1.set_title(u'Number of frequent flyer miles earned per year')
    image1_xlabel_text = image1.set_xlabel(u'Liters of ice cream consumed per week')
    image1_ylabel_text = image1.set_ylabel(u'Percentage of time spent playing video games')
    plt.setp(image1_title_text, size=9, weight='bold', color='red')
    plt.setp(image1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(image1_ylabel_text, size=7, weight='bold', color='black')

    # # 设置图例，这是对图像的说明和解释
    # didtLike = mlines.Line2D([], [], color = 'blue',marker = '.',
    #                             markersize = 10,label = "Did NOt Like" )
    # smallDoses = mlines.Line2D([], [],color = 'black', marker = '.',
    #                             markersize = 10,label = "Liked in Small Doses")
    # largeDoses = mlines.Line2D([], [], color = 'red',marker = '.',
    #                             markersize = 10, label = "Likes in large Doses")
    #
    # # 添加图例
    # image1.legend(handles = [didtLike, smallDoses, largeDoses])
    # plt.show()

    # 设置图例
    didtLike = plt.scatter([], [], color = 'blue',marker= '.',
                           linewidths = 5,label= "did not like")
    smallDoses = plt.scatter([],[], color='black', marker='.',
                            linewidths = 5,label = "liked in small Donses")
    largeDoses = plt.scatter([],[], color='red', marker='.',
                             linewidths = 5,label = "liked in large doses")
    image1.legend(handles = [didtLike, smallDoses, largeDoses])
    plt.show()





# 定义主函数
if __name__ == '__main__':
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    # 调用函数，使得数据可视化
    showDatas(datingDataMat, datingLabels)