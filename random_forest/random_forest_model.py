import numpy as np
from decision_tree.decision_tree_model  import ClassificationTree


class RandomForest():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.

    Parameters:
    -----------
    n_estimators: int
        树的数量
        The number of classification trees that are used.
    max_features: int
        每棵树选用数据集中的最大的特征数
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        每棵树中最小的分割数，比如 min_samples_split = 2表示树切到还剩下两个数据集时就停止
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        每棵树切到小于min_gain后停止
        The minimum impurity required to split the tree further.
    max_depth: int
        每棵树的最大层数
        The maximum depth of a tree.
    """

    def __init__(self, n_estimators=100, min_samples_split=2, min_gain=0,
                 max_depth=float("inf"), max_features=None):

        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.max_features = max_features

        self.trees = []
        # 建立森林(bulid forest)
        for _ in range(self.n_estimators):
            tree = ClassificationTree(min_samples_split=self.min_samples_split, min_impurity=self.min_gain,
                                      max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X, Y):
        # 训练，每棵树使用随机的数据集(bootstrap)和随机的特征
        # every tree use random data set(bootstrap) and random feature
        sub_sets = self.get_bootstrap_data(X, Y)
        n_features = X.shape[1]
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            # 生成随机的特征
            # get random feature
            sub_X, sub_Y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, replace=True)
            sub_X = sub_X[:, idx]
            self.trees[i].fit(sub_X, sub_Y)
            self.trees[i].feature_indices = idx
            print("tree", i, "fit complete")

    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices
            sub_X = X[:, idx]
            y_pre = self.trees[i].predict(sub_X)
            y_preds.append(y_pre)
        y_preds = np.array(y_preds).T
        y_pred = []
        for y_p in y_preds:
            # np.bincount()可以统计每个索引出现的次数
            # np.argmax()可以返回数组中最大值的索引
            # cheak np.bincount() and np.argmax() in numpy Docs
            y_pred.append(np.bincount(y_p.astype('int')).argmax())
        return y_pred

    def get_bootstrap_data(self, X, Y):

        # 通过bootstrap的方式获得n_estimators组数据
        # get int(n_estimators) datas by bootstrap

        m = X.shape[0]
        Y = Y.reshape(m, 1)

        # 合并X和Y，方便bootstrap (conbine X and Y)
        X_Y = np.hstack((X, Y))
        np.random.shuffle(X_Y)

        data_sets = []
        for _ in range(self.n_estimators):
            idm = np.random.choice(m, m, replace=True)
            bootstrap_X_Y = X_Y[idm, :]
            bootstrap_X = bootstrap_X_Y[:, :-1]
            bootstrap_Y = bootstrap_X_Y[:, -1:]
            data_sets.append([bootstrap_X, bootstrap_Y])
        return data_sets