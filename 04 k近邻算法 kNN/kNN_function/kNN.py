import numpy as np
from math import sqrt
from collections import Counter

'''
根据 04-01 中描述的 kNN 的原理和计算过程，
把 kNN 包装成一个可直接调用的函数
'''
def kNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be 1 <= k <= size of X_train"
    assert X_train.shape[0] == y_train.shape[0], "X_train size must equals to y_train size"
    assert X_train.shape[1] == x.shape[0], "the feature number of x must be equal to X_train"

    # 分别计算每个训练样本到待预测点的距离
    distance = [sqrt(sum((x_train - x) ** 2)) for x_train in X_train]

    # 对结果排序，返回按照排序的元素索引注册的数组
    sortDistanceIndex = np.argsort(distance)

    # 从排序索引中取出 top k 的样本的类型值
    topK_y = [y_train[i] for i in sortDistanceIndex[:k]]

    # 按照数组中值做统计, 返回的是一个 Counter 对象， 内部其实是一个 dict， k是被统计的值， v是被统计的值出现的次数
    votes = Counter(topK_y)

    # 调用 Counter 的 most_common 函数拿到出现最多的那个值及其次数
    result = votes.most_common(1)

    # 最后输出预测结果
    predict_y = result[0][0]

    return predict_y
