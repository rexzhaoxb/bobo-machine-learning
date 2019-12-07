""" 常用的度量方法 """
import numpy as np

def accuracy_score(y_test, y_predict):
    """ 通过对预测值和测试集结果的比较，计算模型的准确度 """
    assert y_test.shape[0] == y_predict.shape[0], "y_test size must be equal to y_predict size"
    return sum(y_test == y_predict) / len(y_test)
