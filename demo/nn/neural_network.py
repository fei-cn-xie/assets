import numpy as np  # 导入 NumPy，用于矩阵运算


class NeuralNetwork:
    """
    一个简单的两层神经网络：
        输入层 -> 隐藏层(激活函数 tanh) -> 输出层(激活函数 sigmoid)

    结构示意：
        X  --W1,b1-->  z1  --tanh-->  a1  --W2,b2-->  z2  --sigmoid-->  a2(预测值)
    """

    def __init__(self, n_input, n_hidden, n_output, lr=0.1):
        """
        初始化网络参数
        :param n_input:  输入特征维度（输入层神经元个数）
        :param n_hidden: 隐藏层神经元个数
        :param n_output: 输出维度（输出层神经元个数）
        :param lr:       学习率，控制每次参数更新的步长
        """
        np.random.seed(42)  # 固定随机种子，保证每次运行结果一致，方便调试

        # ---------- 权重和偏置初始化 ----------
        # 权重用小随机数初始化（乘以 0.1 让数值较小，避免激活函数饱和）
        # 不能全部初始化为 0，否则同一层的神经元会学到相同的东西（对称性问题）

        # W1: 输入层 -> 隐藏层 的权重矩阵，形状 (n_input, n_hidden)
        self.W1 = np.random.randn(n_input, n_hidden) * 0.1

        # b1: 隐藏层的偏置，形状 (1, n_hidden)
        # 偏置初始化为 0 是安全的，不会引起对称性问题
        self.b1 = np.zeros((1, n_hidden))

        # W2: 隐藏层 -> 输出层 的权重矩阵，形状 (n_hidden, n_output)
        self.W2 = np.random.randn(n_hidden, n_output) * 0.1

        # b2: 输出层的偏置，形状 (1, n_output)
        self.b2 = np.zeros((1, n_output))

        self.lr = lr  # 保存学习率，训练时使用

    # ---------- 激活函数及其导数 ----------

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid 激活函数：把任意实数压缩到 (0, 1) 区间
        公式：sigmoid(x) = 1 / (1 + e^(-x))
        常用于二分类问题的输出层，输出可解释为概率
        """
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_grad(a):
        """
        Sigmoid 函数的导数
        注意：这里传入的是 sigmoid 的输出 a = sigmoid(z)，而不是 z 本身
        因为 sigmoid 的导数有个很好的性质：sigmoid'(z) = a * (1 - a)
        这样可以省一次计算（不用重新算 sigmoid）
        """
        return a * (1.0 - a)

    # ---------- 前向传播 ----------

    def forward(self, X):
        """
        前向传播：从输入 X 一路计算得到输出 a2（预测值）
        :param X: 输入数据，形状 (m, n_input)，m 是样本数
        :return:  预测值 a2，形状 (m, n_output)
        """
        # ---------- 第 1 步：缓存输入 ----------
        # 反向传播计算梯度时需要用到 X，所以在这里保存下来
        self.X = X

        # ---------- 第 2 步：隐藏层的线性部分 ----------
        # X @ W1：矩阵乘法，(m, n_input) @ (n_input, n_hidden) = (m, n_hidden)
        # + b1：加上偏置，利用 NumPy 的广播机制，把 (1, n_hidden) 广播到每一行
        # z1 是隐藏层的"加权和"，还没经过激活函数
        self.z1 = X @ self.W1 + self.b1

        # ---------- 第 3 步：隐藏层的激活 ----------
        # 对 z1 应用 tanh 激活函数，引入非线性
        # tanh 输出范围 (-1, 1)，比 sigmoid 更常用在隐藏层
        self.a1 = np.tanh(self.z1)

        # ---------- 第 4 步：输出层的线性部分 ----------
        # a1 @ W2：(m, n_hidden) @ (n_hidden, n_output) = (m, n_output)
        # + b2：加上输出层偏置
        self.z2 = self.a1 @ self.W2 + self.b2

        # ---------- 第 5 步：输出层的激活 ----------
        # 用 sigmoid 把输出压缩到 (0, 1)
        # 这里 a2 就是最终的预测值，同时也是反向传播时需要缓存的中间结果
        self.a2 = self.sigmoid(self.z2)

        return self.a2  # 返回预测值

    # ---------- 反向传播 ----------

    def backward(self, y):
        """
        反向传播：根据预测值 a2 和真实标签 y，计算各参数的梯度并更新
        :param y: 真实标签，形状 (m, n_output)
        """
        # 使用的损失函数：均方误差 L = 1/2 * sum((a2 - y)^2)
        # 前面乘 1/2 是为了让求导后更简洁（1/2 的导数是 1，消掉了平方的 2）

        # ============ 输出层梯度 ============
        # 链式法则：dL/dz2 = dL/da2 * da2/dz2
        #   dL/da2 = (a2 - y)（均方误差的导数）
        #   da2/dz2 = sigmoid'(a2) = a2 * (1 - a2)
        # 所以：
        # dz2 表示输出层线性部分的梯度，形状 (m, n_output)
        dz2 = (self.a2 - y) * self.sigmoid_grad(self.a2)

        # dW2 = a1^T @ dz2
        # 因为 z2 = a1 @ W2 + b2，所以对 W2 求导时 a1 在左边
        # 形状：(n_hidden, m) @ (m, n_output) = (n_hidden, n_output)
        # 与 W2 的形状一致，可以直接相减更新
        dW2 = self.a1.T @ dz2

        # db2 = dz2 沿样本维度求和
        # 因为偏置对每个样本都加了同样的值，梯度要累加所有样本
        # keepdims=True 保证结果形状是 (1, n_output)，方便广播
        db2 = dz2.sum(axis=0, keepdims=True)

        # ============ 隐藏层梯度 ============
        # 先把误差从输出层"传回"到隐藏层
        # da1 = dz2 @ W2^T
        # 形状：(m, n_output) @ (n_output, n_hidden) = (m, n_hidden)
        da1 = dz2 @ self.W2.T

        # dz1 = da1 * tanh'(z1)
        # tanh 的导数：tanh'(z) = 1 - tanh(z)^2 = 1 - a1^2
        # 形状：(m, n_hidden)
        dz1 = da1 * (1.0 - self.a1 ** 2)

        # dW1 = X^T @ dz1，原理同 dW2
        # 形状：(n_input, m) @ (m, n_hidden) = (n_input, n_hidden)，与 W1 一致
        dW1 = self.X.T @ dz1

        # db1 = dz1 沿样本维度求和
        # 形状：(1, n_hidden)
        db1 = dz1.sum(axis=0, keepdims=True)

        # ============ 梯度下降更新参数 ============
        # 规则：参数 = 参数 - 学习率 * 梯度
        # 减去梯度是因为我们要"最小化"损失，梯度方向是上升最快的方向
        self.W2 -= self.lr * dW2  # 更新输出层权重
        self.b2 -= self.lr * db2  # 更新输出层偏置
        self.W1 -= self.lr * dW1  # 更新隐藏层权重
        self.b1 -= self.lr * db1  # 更新隐藏层偏置

    # ---------- 训练循环 ----------

    def train(self, X, y, epochs=10000, verbose=True):
        """
        训练网络
        :param X:       训练数据，形状 (m, n_input)
        :param y:       训练标签，形状 (m, n_output)
        :param epochs:  迭代轮数
        :param verbose: 是否打印训练过程
        """
        for i in range(epochs):
            self.forward(X)   # 1. 前向传播，计算预测值（同时缓存中间结果）
            self.backward(y)  # 2. 反向传播，计算梯度并更新参数

            # 每隔 1000 轮打印一次损失，方便观察训练效果
            if verbose and i % 1000 == 0:
                loss = np.mean((self.a2 - y) ** 2)  # 均方误差
                print(f"epoch {i:6d}  loss = {loss:.6f}")

    def predict(self, X):
        """预测：只做前向传播，不更新参数"""
        return self.forward(X)


# ==================== 用 XOR 问题做测试 ====================
if __name__ == "__main__":
    # XOR 问题的输入：4 种组合
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=float)

    # XOR 问题的标签：相同为 0，不同为 1
    # 注意：XOR 是线性不可分的，必须靠隐藏层的非线性激活才能学会
    y = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=float)

    # 创建网络：2 个输入、4 个隐藏神经元、1 个输出，学习率 0.5
    net = NeuralNetwork(n_input=2, n_hidden=4, n_output=1, lr=0.1)

    # 训练 10000 轮
    net.train(X, y, epochs=10000)

    # 打印最终预测结果，round(3) 保留 3 位小数，ravel() 展平成一维
    print("\n预测结果：")
    print(net.predict(X).round(3).ravel())