import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score


class kNNClassifier:
    def __init__(self, k):
        """ 初始化 kNN 分类器 """
        assert k > 1, "k must be positive integer"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to y_train"
        assert X_train.shape[0] >= self.k, "the size of X_train must be more than k"

        self._X_train = X_train
        self._y_train = y_train
        # 参考 scikit-learn 的实现， fit时返回自己
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None, "fit must be call before predict"
        assert self._X_train.shape[1] == X_predict.shape[1], "the X_predict feature number must be equal to X_train"

        """ 之前实现算法一次只能预测一个值，这里参考scikit learn实现可以算多个值，但是底层还是调用一个内部封装好的函数来做 """
        y_predict = [self._predict(x_predict) for x_predict in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        assert self._X_train.shape[1] == x.shape[0], "the x feature number must be equal to X_train"
        """ 下面把 01 中实现的代码拷贝过来 """
        # 分别计算每个训练样本到待预测点的距离
        distance = [sqrt(sum((x_train - x) ** 2)) for x_train in self._X_train]
        # 对结果排序，返回按照排序的元素索引注册的数组
        sortDistanceIndex = np.argsort(distance)
        # 从排序索引中取出 top k 的样本的类型值
        topK_y = [self._y_train[i] for i in sortDistanceIndex[:self.k]]
        # 按照数组中值做统计, 返回的是一个 Counter 对象， 内部其实是一个 dict， k是被统计的值， v是被统计的值出现的次数
        votes = Counter(topK_y)
        # 调用 Counter 的 most_common 函数拿到出现最多的那个值及其次数
        result = votes.most_common(1)
        # 最后输出预测结果
        predict_y = result[0][0]
        return predict_y

    def score(self, X_test, y_test):
        """ 如果只关心模型的准确度，不关心预测的结果，就可以调用这个函数 """
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "kNN(k=%d)" % self.k
