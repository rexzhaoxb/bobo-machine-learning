import numpy as np


class SimpleLinearRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1 and y_train.ndim == 1, "目前只支持一维数据处理"
        assert len(x_train) == len(y_train), "the size of x_train must be equal to y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 分子
        num = 0.0
        # 分母
        d = 0.0

        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, "目前只支持一维数据处理"
        assert self.a_ is not None, "must fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return str.format("SmpleLinearRegression1(a={}, b={})", self.a_, self.b_)


class SimpleLinearRegression2:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1 and y_train.ndim == 1, "目前只支持一维数据处理"
        assert len(x_train) == len(y_train), "the size of x_train must be equal to y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 分子
        num = (x_train - x_mean).dot(y_train - y_mean)
        # 分母
        d = (x_train - x_mean).dot((x_train - x_mean))

        # 循环计算被替换成上面直接用 numpy 的向量点乘(dot)运算
        # for x_i, y_i in zip(x_train, y_train):
        #    num += (x_i - x_mean) * (y_i - y_mean)
        #    d += (x_i - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, "目前只支持一维数据处理"
        assert self.a_ is not None, "must fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return str.format("SmpleLinearRegression2(a={}, b={})", self.a_, self.b_)
