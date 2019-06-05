# sklearn 中关于SVM的使用及方法：
## sklearn.svm.SVC 
    - class sklearn.svm.SVC
    （C = 1.0，kernel ='rbf'，degree = 3，gamma ='auto_deprecated'，
      coef0 = 0.0，shrinking = True，probability = False，tol = 0.001，
      cache_size = 200，class_weight = None，verbose = False，max_iter
       = -1，decision_function_shape =' ovr '，random_state =无）
- C：惩罚项，float类型，可选参数，默认为1.0，C越大，即对分错样本的
     惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，
     也就是对测试数据的分类准确率降低。相反，减小C的话，容许训练样本
     中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，
     一般采用后者，把训练样本集中错误分类的样本作为噪声。
- kernel：核函数类型，str类型，默认为’rbf’。可选参数为： 
    - ’linear’：线性核函数
    - ‘poly’：多项式核函数
    - ‘rbf’：径像核函数/高斯核
    - ‘sigmod’：sigmod核函数
    - ‘precomputed’：核矩阵
    - precomputed 表示自己提前计算好核函数矩阵，这时候算法内部就不
                  再用核函数去计算核矩阵，而是直接用你给的核矩阵，
                  核矩阵需要为n*n的。
                  
- degree：多项式核函数的阶数，int类型，可选参数，默认为3。这个参数
          只对多项式核函数有用，是指多项式核函数的阶数n，如果给的
          核函数参数是其他核函数，则会自动忽略该参数。
          
- gamma：核函数系数，float类型，可选参数，默认为auto。
         只对’rbf’ ,’poly’ ,’sigmod’有效。如果gamma为auto，
         代表其值为样本特征数的倒数，即1/n_features。
         
- coef0：核函数中的独立项，float类型，可选参数，默认为0.0。只有对
          ’poly’ 和,’sigmod’核函数有用，是指其中的参数c。
          
- probability：是否启用概率估计，bool类型，可选参数，默认为False，
         这必须在调用fit()之前启用，并且会fit()方法速度变慢。
         
- shrinking：是否采用启发式收缩方式，bool类型，可选参数，默认为True。

- tol：svm停止训练的误差精度，float类型，可选参数，默认为1e^-3。

- cache_size：内存大小，float类型，可选参数，默认为200。
            指定训练所需要的内存，以MB为单位，默认为200MB。
            
- class_weight：类别权重，dict类型或str类型，可选参数，默认为None。
                给每个类别分别设置不同的惩罚参数C，如果没有给，
                则会给所有类别都给C=1，即前面参数指出的参数C。
                如果给定参数’balance’，则使用y的值自动调整与输入
                数据中的类频率成反比的权重。
                
- verbose：是否启用详细输出，bool类型，默认为False，此设置利用
           libsvm中的每个进程运行时设置，如果启用，可能无法在
           多线程上下文中正常工作。一般情况都设为False，不用管它。
           
- max_iter：最大迭代次数，int类型，默认为-1，表示不限制。

- decision_function_shape：决策函数类型，可选参数’ovo’和’ovr’，
                           默认为’ovr’。’ovo’表示one vs one，
                           ’ovr’表示one vs rest。
                           
- random_state：数据洗牌时的种子值，int类型，可选参数，默认为None。
                伪随机数发生器的种子,在混洗数据时用于概率估计。
                
               
# 总结
## 1 SVM的优缺点
- 优点:
     - 可用于线性/非线性分类，也可以用于回归，泛化错误率低，
       也就是说具有良好的学习能力，且学到的结果具有很好的推广性。
       
     - 可以解决小样本情况下的机器学习问题，可以解决高维问题，
       可以避免神经网络结构选择和局部极小点问题。
       
     - SVM是最好的现成的分类器，现成是指不加修改可直接使用。
       并且能够得到较低的错误率，SVM可以对训练集之外的数据点做很
       好的分类决策。
- 缺点:
        - 对参数调节和和函数的选择敏感。

