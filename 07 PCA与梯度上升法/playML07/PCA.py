import numpy as np

class PCA:
    def __init__(self, n_components):
        assert n_components >= 1, "n_components must be valid >= 1"
        self.n_components = n_components
        self.components_ = None

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components

    def fit(self, X, eta=0.01, n_iters=1e4):
        """获取数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], "n_components must not be greater than feature number of X"

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum(X.dot(w) ** 2) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2.0 / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, init_w, eta, n_iters=1e4, epsilon=1e-8):
            i_iters = 0
            w = direction(init_w)

            while i_iters < n_iters:
                last_w = w
                gradient = df(w, X)
                w = last_w + eta * gradient
                w = direction(w)  # 注意1：每次求一个单位向量
                if abs(f(w, X) - f(last_w, X)) < epsilon:
                    break

                i_iters += 1
            print("i_iters =", i_iters)
            return w

        X_pca = demean(X)
        self.components_ = np.empty((self.n_components, X.shape[1]))
        for i in range(self.n_components):
            init_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, init_w, eta)
            self.components_[i, :] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        """将指定的X，映射到各个主成分分量中"""
        assert self.n_components <= X.shape[1], "n_components must not be greater than feature number of X"
        assert self.components_.shape[1] == X.shape[1], "X features number must be equal to fit sample"

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将低维数据返回高维数据"""
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)