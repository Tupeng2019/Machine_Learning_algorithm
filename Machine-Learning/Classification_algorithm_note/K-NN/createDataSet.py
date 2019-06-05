import numpy as np

# 建立函数，创建数据集，这只是最初的一小步
def createDataSet():
    # 建立几个不同的二维数组，可以是随意的，group
    group = np.array([[1,101],[8,158],[245,9],[112,25]])
    # 对以上的四个二维数组进行标签处理label
    labels = ["不好吃","不好吃","好吃" ,"好吃"]
    # 函数一般都会有返回值
    return group, labels

# 定义主函数
if __name__ == '__main__':
    # 进行数据集的实例化
    group,labels =createDataSet()
    #将我们所定义的数据集进行打印
    print(group)
    print(labels)