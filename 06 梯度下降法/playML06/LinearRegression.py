import numpy as np
from sklearn.metrics import r2_score

class LinearRegression:
    def __init__(self):
        # 系数
        self.coefficients_ = None;
        # 截距
        self.interception_ = None;
        # 把系数和截距放在一起的矩阵
        self._theta = None;

    # 正规方程拟合
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "X_train size must be equal to y_train size"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coefficients_ = self._theta[1:]

        return self

    # 梯度下降拟合
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to y_train"

        # 定义计算损失函数值得方法
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2)
            except:
                return float('inf')
        
        # 定义计算导数值得方法
        def dJ(theta, X_b, y):
        #     result = np.empty(len(theta))
        #     result[0] = np.sum(X_b.dot(theta) - y)
        #     for i in range(1, len(theta)):
        #         result[i] = np.sum((X_b.dot(theta) - y).dot(X_b[:, i]))
        #     return result * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2 / len(X_b)

        # 迭代执行梯度下降过程
        def gradient_descent(X_b, y, init_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = init_theta
            i_iters = 1
            while i_iters <= n_iters:
                # 先记录移动前的参数值（向量）
                last_theta = theta
                # 计算梯度
                gradient = dJ(theta, X_b, y)
                # 往梯度方向移动，计算新的参数值（向量）
                theta = last_theta - eta * gradient
                # 退出机制： 把新旧参数值带入损失函数，
                #               如果两个损失函数的值的差小于一个可接受的误差，即已达到极值点，
                #               否则继续下降
                if (abs(J(last_theta, X_b, y) - J(theta, X_b, y))) < epsilon:
                    break
                i_iters += 1
            print('total steps:', i_iters)
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        init_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, init_theta, eta, n_iters)

        self.interception_ = self._theta[0]
        self.coefficients_ = self._theta[1:]
        return self

    # 改进后的随机梯度，这里的参数 n_iters 是指循环遍历所有样本的次数，所有循环一遍是1
    def fit_sgd(self, X_train, y_train, n_iters=3, t0=5, t1=50):
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to y_train"
        assert n_iters > 0

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i)

        def sgd(X_b, y, init_theta, n_iters):

            def learning_rate(t):
                return t0 / (t1 + t)

            theta = init_theta
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(len(X_b))
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(len(X_b)):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * len(X_b) + i) * gradient

            print('total steps:', n_iters + len(X_b))
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        init_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, init_theta, n_iters)

        self.interception_ = self._theta[0]
        self.coefficients_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        assert self.coefficients_ is not None, "must fit before predict"
        assert X_predict.shape[1] == len(self.coefficients_), "feature numbers of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"