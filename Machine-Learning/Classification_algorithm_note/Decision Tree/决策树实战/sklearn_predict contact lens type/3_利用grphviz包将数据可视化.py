from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn.tree import export_graphviz
import pandas as pd
import numpy as np
import pydotplus
import graphviz



if __name__ == '__main__':
    # 加载文件
    with open('lenses.txt', 'r') as fr:
        # 处理文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 提取每组数据的类型，保存在新的列表中
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    # 数据集的特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # # 保存lenses数据的临时列表
    lenses_list = []
    # 创建一个字典，用于保存lenses数据的，用于pandas数组
    lenses_dict = {}
    # 提取信息，对应于生成字典，就相当于每一个标签对应于一列数据
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # 打印处理后的字典信息，在这里不需要打印
    # print(lenses_dict)
    # 生成pandas数据结构，DataFrame
    # lenses_pd 的数据就是有序的数据了
    lenses_pd = pd.DataFrame(lenses_dict)
    # 打印pandas.Dataframe
    print(lenses_pd)
    # 将LabelEncoder（）实例化，即创建对象，用于序列化
    le = LabelEncoder()
    # columns= 列，这就是将每一列都进行序列化
    # 就是将标签下的数据进行数字化，用0,1,2来表示
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # 最终打印该序列化后的数据,
    #print(lenses_pd)  # 这里是不需要打印的

    # 创建DecisionTreeClasstfier（）类，这里得参数，在前的文档中都有介绍：
    # max_depth= 4,bianshi 就是表示的决策树的深度就是4层
    clf = tree.DecisionTreeClassifier(max_depth=4)
    # 使用数据集，构建决策树
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    dot_data = StringIO()
    # 使用graphviz模块绘制决策树，实现决策树的可视化
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # 保存沪指好的决策树，以PDF形式存储
    graph.write_pdf("tree.pdf")
    print('*' * 50)
    print(clf.predict([[1, 1 ,1, 0]]))

