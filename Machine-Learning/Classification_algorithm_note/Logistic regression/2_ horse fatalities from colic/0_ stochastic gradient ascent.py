


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 返回dataMatrix的大小。m为行数,n为列数。
    m,n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 降低alpha的大小，每次减小1/(j+i)。
            # 相比如前面所做的改变：Alpha在每一次迭代中，都进行了改变
            alpha = 4/(1.0+j+i)+0.01
            # 随机选取样本
            # 这里也是做出的改变：是随机的选择样本，而不是遍历所有的样本
            randIndex = int(random.uniform(0,len(dataIndex))
            # 随机选取的一个样本，计算h
            # h就是代表z的含义，梯度上升矢量化公式
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 将已经使用过得样本进行删除
            del(dataIndex[randIndex])
    # 返回权重
    return weights