
import os
import jieba

def TextProcessing(folder_path):
    # 查看folder_path下的文件
    folder_list = os.listdir(folder_path)
    # 建立储存训练集的列表
    data_list = []
    class_list = []

    #遍历每个子文件夹
    for folder in folder_list:
        # 根据子文件夹，生成新的路径
        new_folder_path = os.path.join(folder_path, folder)
        # 根据子文件夹，生成新的路径
        files = os.listdir(new_folder_path)
        j = 1
        #遍历每个txt文件
        for file in files:
            # 每类txt样本数最多100个
            if j > 100:
                break
            # 打开txt文件
            with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:
                raw = f.read()
            # 精简模式，返回一个可迭代的generator
            word_cut = jieba.cut(raw, cut_all = False)
            # generator转换为list
            word_list = list(word_cut)
            data_list.append(word_list)
            class_list.append(folder)
            j += 1
        print(data_list)
        print(class_list)
if __name__ == '__main__':
    #文本预处理
    # 训练集存放地址
    folder_path = './Sample'
    TextProcessing(folder_path)