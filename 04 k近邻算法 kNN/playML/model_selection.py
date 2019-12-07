"""
测试机器学习算法，才能选择出最佳的模型，此模块的函数就是为了解决模型选择中的一些通用问题。
"""

import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], "X size must equal to y"
    assert 0 < test_ratio < 1, "test_ratio must great than 0, and less than 1"

    # 有时候测试需要反复用同一套随机数，提供 seed 作为可选参数，来产生可重复出现的随机数
    if seed:
        np.random.seed(seed)

    # 把索引进行随机打乱排列
    shuffled_indexes = np.random.permutation(len(X))

    # 设置抽取测试数据集的比例
    test_ratio = 0.2
    test_size = int(len(X) * test_ratio)

    # 分别抽取训练数据集和测试数据集
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test

