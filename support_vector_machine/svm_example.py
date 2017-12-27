# coding:utf-8

import logging

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification

from pca.pca import *
from support_vector_machine.kernels import *
from support_vector_machine.svmModel import *
logging.basicConfig(level=logging.DEBUG)

import time
def run():
    start = time.clock()
    X, y = make_classification(n_samples=1200, n_features=10, n_informative=5,
                               random_state=1111, n_classes=2, class_sep=1.75, )
    # y的标签取值{0,1} 变成 {-1, 1}
    y = (y * 2) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1111)


    #这里是用高斯核，可以用线性核函数和多项式核函数
    kernel = RBF(gamma=0.1)
    model = SVM(X_train,y_train,max_iter=500, kernel=kernel, C=0.6)
    model.train()

    predictions = model.predict(X_test)

    accuracyRate = accuracy(y_test, predictions)

    print('Classification accuracy (%s): %s'
          % (kernel, accuracyRate))


    #原来的数据的呈现
    #plot_in_2d(X_test, y_test, title="Support Vector Machine", accuracy=accuracyRate)

    #分类的效果
    plot_in_2d(X_test, predictions, title="Support Vector Machine", accuracy=accuracyRate)

    end = time.clock()
    print("read: %f s" % (end - start))



if __name__ == '__main__':
    run()

