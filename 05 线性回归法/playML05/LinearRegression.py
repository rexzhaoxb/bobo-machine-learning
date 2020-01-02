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

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "X_train size must be equal to y_train size"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
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