
import os
import random
import jieba

"""
函数说明:中文文本处理

Parameters:
    folder_path - 文本存放的路径
    test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
    all_words_list - 按词频降序排序的训练集列表
    train_data_list - 训练集列表
    test_data_list - 测试集列表
    train_class_list - 训练集标签列表
    test_class_list - 测试集标签列表

"""
# 该函数就是对中文文本的处理
def TextProcessing(folder_path, test_size = 0.2):
    # 查看folder_path下的路径的文件
    folder_list = os.listdir(folder_path)
    # 创建空列表放数据集数据
    data_list = []
    # 创建空列表放数据集类型
    class_list = []

    #遍历每个子文件夹
    for folder in folder_list:
        # 根据子文件夹，生成新的路径
        new_folder_path = os.path.join(folder_path, folder)
        # 存放子文件夹下的TXT文件的列表
        files = os.listdir(new_folder_path)
        # 定义j为每一类TXT样本的个数，并初始化为1
        j = 1
        #遍历每个txt文件
        for file in files:

            #每一类样本的个数最多为100个
            if j > 100:
                break
            # 打开txt文件
            with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:
                raw = f.read()
            #  精简模式，返回一个可迭代的generator
            word_cut = jieba.cut(raw, cut_all = False)
            # 将generator装换为list
            word_list = list(word_cut)

            # 添加数据集数据
            data_list.append(word_list)
            # 向列表中添加数据集类型
            class_list.append(folder)
            j += 1
    # #zip压缩合并，将数据与标签对应压缩
    data_class_list = list(zip(data_list, class_list))
    # 将data_class_list乱序
    random.shuffle(data_class_list)
    # 训练集和测试集切分的索引值
    index = int(len(data_class_list) * test_size) + 1
    # 训练集
    train_list = data_class_list[index:]
    # 测试集
    test_list = data_class_list[:index]
    # 训练集解压缩
    train_data_list, train_class_list = zip(*train_list)
    # 测试集解压缩
    test_data_list, test_class_list = zip(*test_list)
    # 统计训练集词频
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    #根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse = True)
    # 解压缩
    all_words_list, all_words_nums = zip(*all_words_tuple_list)
    # 将all_words_list转换成列表
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

# 读取文件中的内容，并且去重
def MakeWordsSet(words_file):
    # 创建set集合
    words_set = set()
    # 打开文件夹
    with open(words_file, 'r', encoding = 'utf-8') as f:
        # 以行为单位进行一行一行的读取
        for line in f.readlines():
            # 去掉回车，就是读取为一行
            word = line.strip()
            # 如果有文本，就添加到words_set中去
            if len(word) > 0:
                words_set.add(word)
    return words_set

'''
# 文本的特征提取：
# all_words_list就是训练集所有的文本
# deleteN 删除词频最高的单词
stopwords 指定的结束语

'''
def words_dict(all_words_list, deleteN, stopwords_set = set()):
    # 特征列表
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        # feature_words 的维度为1000
        if n > 1000:
            break
        #如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

if __name__ == '__main__':
    #文本预处理
    # 训练集存放地址
    folder_path = './Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    print(all_words_list)

