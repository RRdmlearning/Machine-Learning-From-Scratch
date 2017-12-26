# coding:utf-8
import numpy as np
import scipy.spatial.distance as dist


#线性核函数
class LinearKernel(object):
    def __call__(self, x, y):
        return np.dot(x, y.T)


#多项式核函数
class PolyKernel(object):
    #初始化方法
    def __init__(self, degree=2):
        self.degree = degree

    def __call__(self, x, y):
        return np.dot(x, y.T) ** self.degree

#高斯核函数
class RBF(object):
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def __call__(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        return np.exp(-self.gamma * dist.cdist(x, y) ** 2).flatten()
