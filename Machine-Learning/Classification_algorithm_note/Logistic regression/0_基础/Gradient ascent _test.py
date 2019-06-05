



'''
函数说明:梯度上升算法测试函数
求函数f(x) = -x^2 + 4x的极大值

'''
def Gradient_Ascent_test():
    # 表示f（x）的导数
    def f_prime(x_old):
        # 这就是我们所要计算的公式
        return -2 * x_old + 4
    # 初始值，给一个小于x_new的值
    x_old = -1
    # 梯度上升算法的初始值，即从（0，0）开始
    x_new = 0
    # 步长就是学习速率，控制更新的快慢
    alpha = 0.01
    # 精度，就是更新的阈值
    presision = 0.000001
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)

if __name__ == '__main__':
    Gradient_Ascent_test()
