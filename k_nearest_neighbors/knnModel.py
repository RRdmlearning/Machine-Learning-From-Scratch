# coding:utf-8
import numpy as np
from mlfromscratch.utils import euclidean_distance


class KNN():

    def __init__(self,k=5):

        self.k = k





    def predict(self, X_test, X_train, y_train):

        y_predict = np.zeros(X_test.shape[0])

        for i in range(X_test.shape[0]):

            distances = np.zeros((X_train.shape[0], 2)) #测试的数据和训练的各个数据的欧式距离

            for j in range(X_train.shape[0]):
                dis = euclidean_distance(X_test[i], X_train[j]) #计算欧式距离
                label = y_train[j] #测试集到的每个训练集的数据的分类标签
                distances[j] = [dis, label]

                # argsort()得到测试集到训练的各个数据的欧式距离从小到大排列并且得到序列，然后再取前k个.
                k_nearest_neighbors = distances[distances[:, 0].argsort()][:self.k]

                #利用np.bincount统计k个近邻里面各类别出现的次数
                counts = np.bincount(k_nearest_neighbors[:, 1].astype('int'))

                #得出每个测试数据k个近邻里面各类别出现的次数最多的类别
                testLabel = counts.argmax()
                y_predict[i] = testLabel

        return y_predict


