import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self):
        # 系数
        self.coefficients_ = None;
        # 截距
        self.interception_ = None;
        # 把系数和截距放在一起的矩阵
        self._theta = None;

    def _sigmoid(self, t):
        return 1 / (1 + np.exp(-t))


    # 由于逻辑回归没有对应的数学公式，所以删掉了正规方程的 fit
    # 由于逻辑回归最后推导的损失函数是一个凸函数，所以不需要随机梯度 fit，也删掉

    # 梯度下降拟合
    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to y_train"

        # 定义计算损失函数值得方法
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')
        
        # 定义计算导数值得方法
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

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

    def predict_proba(self, X_predict):
        assert self.coefficients_ is not None, "must fit before predict"
        assert X_predict.shape[1] == len(self.coefficients_), "feature numbers of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"