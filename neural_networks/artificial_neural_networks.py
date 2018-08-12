# -*-coding:utf-8-*-
"""
@Time   : 18/8/12 下午7:23
@Author : Mark
@File   : artificial_neural_networks.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

plt.ion()
np.random.seed()
X, y = make_moons(200, noise=0.20)

num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2

# 梯度下降参数
epsilon = 0.01
reg_lambda = 0.01


def plot_descision_boundary(pred_func):
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 画图
    plt.contour(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# 定义损失函数
def calculate_loss(model):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    # 向前推进，前向运算
    # 第一个隐层
    z1 = X.dot(W1) + b1
    # 激活函数
    a1 = np.tanh(z1)
    # 第二个隐层
    z2 = a1.dot(W2) + b2
    # 指数运算
    exp_scores = np.exp(z2)
    # 概率
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 计算损失
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # 正则化
    data_loss += reg_lambda / 2 * (np.square(W1) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


# 完整的训练建模函数定义
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    '''
    参数：
    1.nn_hdim:隐层节点个数
    2.num_passes:梯度下降迭代次数
    3.print_loss:设定为True的话，每1000次迭代输出一次loss的当前值
    '''

    # 随机初始化权重
    np.random.seed(0)
    W1 = np.random.rand(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.rand(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # 最后学到的模型
    model = {}
    # 开始梯度下降
    for i in xrange(0, num_passes):
        # 向前计算loss
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 反向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 加上正则化项
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # 梯度下降更新参数
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # 得到的模型其实是这些权重
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # 打印中间过程
        # if print_loss and i % 1000 == 0:
        #     print 'Loss after interation %i: %f' % (i, calculate_loss(model))
    return model


# 定义判定结果的函数
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 前向运算
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)

    # 计算概率输出最大概率对应的类别
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# 建立3个隐层的神经网络
model = build_model(3, print_loss=True)

# 画出决策/判定边界
plot_descision_boundary(lambda x: predict(model, x))
plt.title('Decision Boundary for hidden layer size 3')
plt.show()

# 不同的隐层对结果的影响
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_descision_boundary(lambda x: predict(model, x))
plt.show()
