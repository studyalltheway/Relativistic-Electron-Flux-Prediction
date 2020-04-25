# gradient information
# MLP based model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import RMSE, PE, NORM
from model import create_model2
from preprocess import dataprocess1, dataprocess2
import tensorflow as tf
from config import config
from supermodel import supermodel1
import multiprocessing
import gc
gc.disable()

import time
T0 = time.time()


def NORM2(data):
    data = np.array(data).reshape(-1)
    datasum = np.sum(data)
    data = data/datasum
    return data


if __name__=='__main__':
    F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,Bx,By,Bz,E,Kp,Dst,ap,AE = dataprocess1()
    data_in = np.hstack((F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,Bx,By,Bz,E,Kp,Dst,ap,AE))
    print('Modified data_in shape:', data_in.shape)
    data_out, _ = dataprocess2()
    print('data_out shape:', data_out.shape)

    Setpre = data_in
    Labelpre = data_out
    config = config()

    #pool = multiprocessing.Pool(processes=7)
    ######-------------------------------------七年模型分年训练
    oneyear = 343
    yrange = np.arange(oneyear)
    ShowSet = Setpre[yrange, :]
    ShowLabel = Labelpre[yrange, :]
    Set = np.delete(Setpre, yrange, axis=0)
    Label = np.delete(Labelpre, yrange, axis=0)
    trainset = Set
    trainlabel = Label
    Batch_size = 200
    ###--------------
    num_in = 100
    x = tf.placeholder(tf.float32, [None, num_in])
    y0 = tf.placeholder(tf.float32, [None, 1])
    y1, totalloss, loss, a, a2 = create_model2( \
        x, y0, num_in, reg2=0.001, node1=25, node2=25)

    gradient_nodes1 = tf.gradients(y1, x)
    gradient_nodes2 = tf.gradients(loss, x)
    gradient_nodes3 = tf.gradients(totalloss, x)
    gradient_nodes8 = tf.gradients(y1, a)
    gradient_nodes9 = tf.gradients(y1, a2)

    learningrate_base = 0.01
    learningrate_decay = 0.999
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learningrate_base,
        global_step,
        1500 // Batch_size,
        learningrate_decay,
        staircase=True
    )
    trainstep = tf.train.AdamOptimizer(learning_rate).minimize(totalloss, global_step=global_step)
    m_saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        X = trainset
        Y = trainlabel
        for ii in range(1500):
            start = (ii * Batch_size) % len(trainlabel)
            end = start + Batch_size
            sess.run(trainstep, feed_dict={x: X[start:end], y0: Y[start:end]})
            if ii%100==0:
                print('Loss:', sess.run(loss, feed_dict={x: X[start:end], y0: Y[start:end]}))
                print('Total Loss:', sess.run(totalloss, feed_dict={x: X[start:end], y0: Y[start:end]}))
                #print('Gradient nodes:', sess.run(gradient_nodes, feed_dict={x: X, y0: Y}))
        trainloss = sess.run(loss, feed_dict={x: X, y0: Y})
        grad_res1 = sess.run(gradient_nodes1, feed_dict={x: X, y0: Y})
        grad_res2 = sess.run(gradient_nodes2, feed_dict={x: X, y0: Y})
        grad_res3 = sess.run(gradient_nodes3, feed_dict={x: X, y0: Y})
        grad_res8 = sess.run(gradient_nodes8, feed_dict={x: X, y0: Y})
        grad_res9 = sess.run(gradient_nodes9, feed_dict={x: X, y0: Y})
        print('Gradient nodes:\n', grad_res1)
        print('Gradient nodes 2:', grad_res2)
        print('total_loss1:', trainloss)

    grad_res1_2 = np.array(grad_res1[0])  #grad_res1_2表示不采用绝对值计算得到的结果
    grad_res1_2 = np.mean(grad_res1_2, axis=0)
    grad_res1 = np.array(grad_res1[0])
    grad_res1 = np.mean(np.abs(grad_res1), axis=0)
    grad_res2 = np.array(grad_res2[0])
    grad_res2 = np.mean(np.abs(grad_res2), axis=0)
    grad_res3 = np.array(grad_res3[0])
    grad_res3 = np.mean(np.abs(grad_res3), axis=0)
    grad_res8 = np.array(grad_res8[0])
    grad_res8 = np.mean(np.abs(grad_res8), axis=0)
    grad_res9 = np.array(grad_res9[0])
    grad_res9 = np.mean(np.abs(grad_res9), axis=0)
    print('grad_res1:\n', grad_res1)
    print('grad_res2:\n', grad_res2)
    print('grad_res3:\n', grad_res3)
    print('grad_res8:\n', grad_res8)
    print('grad_res9:\n', grad_res9)
    g1_2 = NORM2(grad_res1_2)
    g1 = NORM2(grad_res1)
    g2 = NORM2(grad_res2)
    g3 = NORM2(grad_res3)
    g8 = NORM2(grad_res8)
    g9 = NORM2(grad_res9)
    xx = np.arange(100)
    fig = plt.figure(figsize=(20, 12))
    plt.legend(['g1', 'w', 'g7'])
    plt.show()
    fig2 = plt.figure(figsize=(20, 12))
    xx = np.arange(25)
    plt.plot(xx, g8, 'ro--')
    plt.plot(xx, g9, 'bo--')
    plt.show()

    xx = np.arange(100)
    fig3 = plt.figure(figsize=(20, 12))
    plt.plot(xx, grad_res1, 'ro-', markersize=5)
    plt.legend(['g1', 'w', 'g7'])
    plt.show()

    fig4 = plt.figure(figsize=(20, 12))
    plt.plot(xx, g1_2, 'ro-', markersize=5)
    plt.plot(xx, grad_res1_2, 'ro--', markersize=5)
    plt.plot(xx, g1, 'bo-', markersize=5)
    plt.plot(xx, grad_res1, 'bo--', markersize=5)
    plt.show()









