#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: SpaceSkyNet
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import os, random

# 保证 alpha 在 [L, H] 内
def clip_alpha(alpha, H, L): 
    if alpha < L: return L
    if alpha > H: return H
    return alpha

# 核函数，径向基函数 (radial bias function)
def rbf_ker(X, A, k = 1.3):
    m = np.shape(X)[0]
    K = np.mat(np.zeros((m, 1)))
    for j in range(m):
        delta_row = X[j, :] - A
        K[j] = delta_row * delta_row.T
    K = np.exp(K / (-1 * k ** 2))
    return K

class rbtSVM(object):
    def __init__(self, data_characteristics = None, labels = None, C = 200, toler =  0.0001, k = 1.3):  # 存储各类参数
        self.train_X = np.mat(data_characteristics)  # 数据特征
        self.labels = np.mat(labels).transpose()  # 数据类别
        self.C = C  # 软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler  # 停止阀值
        self.m = np.shape(data_characteristics)[0] if data_characteristics else 0 # 数据行数
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0  # 初始设为0
        self.error_cache = np.mat(np.zeros((self.m, 2)))  # 缓存
        self.K = np.mat(np.zeros((self.m, self.m)))  # 核函数的计算结果
        for i in range(self.m): self.K[:, i] = rbf_ker(self.train_X, self.train_X[i, :], k)

    def calc_error(self, k):
        return float(np.multiply(self.alphas, self.labels).T * self.K[:, k] + self.b) - float(self.labels[k])

    # 随机选取 alpha_j，并返回 error_j
    def select_alpha_j(self, i, error_i):
        max_k, max_delta_error, error_j = -1, 0, 0
        self.error_cache[i] = [1, error_i]
        candidate_alpha_list = np.nonzero(self.error_cache[:, 0].A)[0]  # 返回矩阵中的非零位置的行数
        if len(candidate_alpha_list) > 1:
            for k in candidate_alpha_list:
                if k == i: continue
                error_k = self.calc_error(k)
                delta_error = abs(error_i - error_k)
                if (delta_error > max_delta_error):  # 返回步长最大的 alpha_j
                    max_k, max_delta_error, error_j = k, delta_error, error_k
            return max_k, error_j
        else:
            # 在0-m中随机选择一个不是i的整数
            j = random.choice(list(set([x for x in range(0, self.m + 1)]) - set([i])))
            error_j = self.calc_error(j)
            return j, error_j

    # 更新数据
    def update_error(self, k):
        error = self.calc_error(k)
        self.error_cache[k] = [1, error]

    def inner_loop(self, i):
        error_i = self.calc_error(i)

        # 检验这行数据是否符合KKT条件，如果不满足，随机选择 alpha_j 进行优化，并更新 alpha_i , alpha_j, b 的值
        if self.labels[i] * error_i < -self.tol and self.alphas[i] < self.C or \
            self.labels[i] * error_i > self.tol and self.alphas[i] > 0:

            # step 1: 选取 alpha_j
            j, error_j = self.select_alpha_j(i, error_i)  # 随机选取aj，并返回其Error值
            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()
            
            # step 2: 计算 alpha_j 的边界
            if self.labels[i] != self.labels[j]:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                return 0
            
            # step 3: 计算 eta, 即样本i，j的相似度
            eta = 2.0 * self.K[i, j] - self.K[i, i] \
                    - self.K[j, j]
            if eta >= 0: return 0

            # step 4: 更新并控制 alpha_j 的取值在 [L, H] 间
            self.alphas[j] -= self.labels[j]*(error_i - error_j) / eta
            self.alphas[j] = clip_alpha(self.alphas[j], H, L)
            self.update_error(j)

            # step 5: alpha 变化大小是否大于阀值，否则返回
            if (abs(self.alphas[j] - alpha_j_old) < self.tol): return 0
            
            # step 6: 最优化 alpha_j 后更新 alpha_i
            self.alphas[i] += self.labels[j]*self.labels[i] * (alpha_j_old - self.alphas[j])
            self.update_error(i) 
                    
            # step 7: 更新 b
            b_i = self.b - error_i - self.labels[i] * (self.alphas[i] - alpha_i_old) * \
                self.K[i, i] - self.labels[j] * (self.alphas[j] - alpha_j_old) * self.K[i, j]
            b_j = self.b - error_j - self.labels[i] * (self.alphas[i] - alpha_i_old) * \
                self.K[i, j] - self.labels[j] * (self.alphas[j] - alpha_j_old) * self.K[j, j]
            if 0 < self.alphas[i] and self.alphas[i] < self.C:
                self.b = b_i
            elif 0 < self.alphas[j] and self.alphas[j] < self.C:
                self.b = b_j
            else:
                self.b = (b_i + b_j) / 2.0
            
            return 1
        else:
            return 0

    # SMO函数，用于快速求解出 alpha, b
    def SMO(self, max_iter = 10000):
        iter_count, alpha_pairs_changed, entire_set = 0, 0, True
        while (iter_count < max_iter) and ((alpha_pairs_changed > 0) or (entire_set)):
            alpha_pairs_changed = 0
            if entire_set:
                # 遍历所有数据
                for i in range(self.m):
                    alpha_pairs_changed += self.inner_loop(i)
                    print("\tfull, iter: %d i = %d, %d pairs changed." % (iter_count, i, alpha_pairs_changed))
                iter_count += 1
            else:
                non_bound_alphas_list  = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                # 遍历非边界的数据
                for i in non_bound_alphas_list :
                    alpha_pairs_changed += self.inner_loop(i)
                    print("\tnon-bound, iter: %d i = %d, %d pairs changed." % (iter_count, i, alpha_pairs_changed))
                iter_count += 1
            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True
            print("iteration number: %d" % iter_count)
        pass

    def fit(self, data_characteristics = None, labels = None):
        if data_characteristics and labels:
            self.__init__(data_characteristics, labels)
        self.SMO()
        pass
    
    def predict(self, test_X = None):
        support_vectors_index = np.nonzero(self.alphas)[0]
        support_vectors = self.train_X[support_vectors_index]
        support_vector_labels = self.labels[support_vectors_index]
        support_vector_alphas = self.alphas[support_vectors_index]

        print("There are %d Support Vectors" % np.shape(support_vectors)[0])

        # 若无测试数据，则用训练数据做测试
        if test_X: test_X = np.mat(test_X)
        else: test_X = self.train_X
        m = np.shape(test_X)[0]
        prediction = []
        for i in range(m):
            kernel_eval = rbf_ker(support_vectors, test_X[i, :])
            prediction.append(int(np.sign(kernel_eval.T * np.multiply(support_vector_labels, support_vector_alphas) + self.b)))
        return prediction

if __name__ == "__main__":
    # 读取前两种鸢尾花（setosa, versicolor）的数据
    iris = pd.read_csv(os.path.join("data", "iris.data"),
                       names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"], nrows=100)
    # 载入特征集和标签集
    X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = iris['Species']

    # 将标签集编码成 1, -1, 1: Iris-setosa, -1: Iris-versicolor
    Y = np.array(list(map(lambda s: 1 if s == 'Iris-setosa' else -1, Y)))
    # print(X, Y)

    # 拆分训练数据和测试数据，7 : 3
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.3, random_state = 2333)
    # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    
    # radial bias function SVM 模型准确度检验
    model = rbtSVM()
    model.fit(train_X.values.tolist(), train_Y.tolist())

    # train data
    prediction = model.predict()
    print('The accuracy of train data is: {0}'.format(metrics.accuracy_score(prediction, train_Y)))

    # test data
    prediction = model.predict(test_X.values.tolist())
    print('The accuracy of test data by the radial bias function SVM is: {0}'.format(metrics.accuracy_score(prediction, test_Y)))

    # sklearn 内置 svm 模型准确度检验
    model = svm.SVC()
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    print('The accuracy of the sklearn SVC SVM is: {0}'.format(metrics.accuracy_score(prediction, test_Y)))