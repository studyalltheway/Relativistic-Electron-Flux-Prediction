#  gradient information, hessian matrix information
# from process4_2, Feedforward neural network
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from model import create_model3
from preprocess import dataprocess1, dataprocess2
import matplotlib.ticker as ticker
import tensorflow as tf
from config import config as config2
import multiprocessing
import time
T0 = time.time()


# calculate the contribution of the first order increments
def ratecal(data1, data2):
    """
    :param fit:
    :param label:
    :return:
    """
    s1, s2 = 0, 0
    for i in range(len(data1)):
        s1 += abs(data2[i]-data1[i])
        s2 += abs(data2[i])
    return 1-s1/s2


# normalization of RI
def NORM2(data):
    data = np.array(data).reshape(-1)
    datasum = np.sum(data)
    data = data/datasum
    return data


# first order
def firstorder(Set, Label, ShowSet, ShowLabel, config):
    trainset = Set
    trainlabel = Label

    # training set shuffle
    data2 = np.hstack((trainset, trainlabel))
    np.random.seed(2020)  # This year is 2020
    np.random.shuffle(data2)
    trainset = data2[:, 0:-1]
    trainlabel = data2[:, -1].reshape(-1, 1)

    if hasattr(config, 'seed'):
        seed = config.seed
    if hasattr(config, 'batch_size'):
        batch_size = config.batch_size
    if hasattr(config, 'num_in'):
        num_in = config.num_in
    if hasattr(config, 'reg1'):
        reg1 = config.reg1
    if hasattr(config, 'reg2'):
        reg2 = config.reg2
    if hasattr(config, 'node1'):
        node1 = config.node1
    if hasattr(config, 'node2'):
        node2 = config.node2
    if hasattr(config, 'weight_init'):
        weight_init = config.weight_init
    if hasattr(config, 'learnrate'):
        learnrate = config.learnrate
    if hasattr(config, 'epochs'):
        epochs = config.epochs

    # build model
    x = tf.placeholder(tf.float32, [None, num_in])
    y0 = tf.placeholder(tf.float32, [None, 1])
    y1, totalloss, loss, inputweight, x1, a, a2 = create_model3( \
        x, y0, num_in, reg1=reg1, reg2=reg2, node1=node1, node2=node2, seed=seed, weight_init=weight_init)

    #params = tf.trainable_variables('Variable:0')
    #print('params:', params)
    gradient_nodes1 = tf.gradients(y1, x)
    #gradient_nodes2 = tf.hessians(y1, params[0])

    # Adam optimizer
    learningrate_base = learnrate
    learningrate_decay = 0.999
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learningrate_base,
        global_step,
        epochs // batch_size,
        learningrate_decay,
        staircase=False
    )
    trainstep = tf.train.AdamOptimizer(learning_rate).minimize(totalloss, global_step=global_step)

    # training
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        X = trainset
        Y = trainlabel
        for ii in range(epochs):
            start = (ii * batch_size) % len(trainlabel)
            end = start + batch_size
            sess.run(trainstep, feed_dict={x: X[start:end], y0: Y[start:end]})
            # if ii%100==0:
            #     print('Loss:', sess.run(loss, feed_dict={x: X[start:end], y0: Y[start:end]}))
            #     print('Total Loss:', sess.run(totalloss, feed_dict={x: X[start:end], y0: Y[start:end]}))
        #trainloss = sess.run(loss, feed_dict={x: X, y0: Y})
        DF1, DF2, DY, DY2, DG, DDX = [], [], [], [], [], []
        XX, YY = ShowSet, ShowLabel
        for i in range(XX.shape[0]-1):
            dx = XX[i+1, :]-XX[i, :]
            grad_res1 = sess.run(gradient_nodes1, feed_dict={x: XX[i, :].reshape(1, 100)})
            grad_res1_2 = sess.run(gradient_nodes1, feed_dict={x: XX[i+1, :].reshape(1, 100)})
            grad_res1 = (np.array(grad_res1)+np.array(grad_res1_2))/2
            df1 = grad_res1.reshape(100)*dx.reshape(100)
            DF1.append(df1)
            # grad_res2 = sess.run(gradient_nodes2, feed_dict={x: X[i, :].reshape(1, 100)})
            # grad_res2 = np.array(grad_res2)
            # if i == 0:
            #     print('grad shape:', grad_res2.shape)
            # dx2 = np.array(dx).reshape(100, 1)
            # dx3 = dx2*(dx2.reshape(1, 100))
            # df2 = np.multiply(dx3, grad_res2[0])
            # DF2.append(df2)
            Y2 = sess.run(y1, feed_dict={x: XX[i+1, :].reshape(1, 100)})
            Y1 = sess.run(y1, feed_dict={x: XX[i, :].reshape(1, 100)})
            DY.append(Y2[0][0]-Y1[0][0])
            dxx = YY[i+1][0] - YY[i][0]
            DY2.append(dxx)
    return DF1, DY, DY2  # first order increments; increments of output; real increments


def main2(config):
    F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,Bx,By,Bz,E,Kp,Dst,ap,AE = dataprocess1()
    data_in = np.hstack((F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,Bx,By,Bz,E,Kp,Dst,ap,AE))
    print('Modified data_in shape:', data_in.shape)
    data_out = dataprocess2()
    print('data_out shape:', data_out.shape)

    data_in = data_in #/10
    Setpre = data_in
    Labelpre = data_out

    ## seven years data, split into 7 parts.  2408 = 344*7
    oneyear = 344
    DF1, Dy, Dy2 = [], [], []

    for i in range(7):
        print('The %d year'%i)
        yrange = np.arange(oneyear*i, oneyear*i+oneyear)
        ShowSet = Setpre[yrange, :]
        ShowLabel = Labelpre[yrange, :]
        Set = np.delete(Setpre, yrange, axis=0)
        Label = np.delete(Labelpre, yrange, axis=0)
        df1, dy, dy2 = firstorder(Set, Label, ShowSet, ShowLabel, config)
        DF1.append(df1)
        Dy.append(dy)
        Dy2.append(dy2)

    DF1 = np.array(DF1).reshape(-1, 100)
    Dy = np.array(Dy).reshape(-1)
    Dy2 = np.array(Dy2).reshape(-1)
    print('D shape:', DF1.shape, Dy.shape, Dy2.shape)

    fx1 = np.sum(np.abs(np.array(DF1)), axis=0)
    fx1 = NORM2(fx1)

    Dx = list(np.sum(np.array(DF1), axis=1).reshape(-1))
    print('Dx:', Dx)
    rate1 = ratecal(Dx, Dy)
    rate1_2 = ratecal(Dx, Dy2)
    print('rate1', rate1, rate1_2)
    # fisrt order information
    fx1 = np.array(fx1).reshape(-1)
    return fx1





if __name__=='__main__':
    # multiprocess
    pool = multiprocessing.Pool(processes=7)

    res, res2 = [], []
    for i in range(20):
        config = config2()
        config.seed += i
        res.append(pool.apply_async(main2, (config,)))  # must be tuple, so (config, )  "," is a must
        print('>>>>>>>\n>>>>>>>>>>>', i)
    pool.close()
    pool.join()
    print('finished !!!')
    print('res!!!', res)
    for j in res:
        x = j.get()
        res2.append(x)

    res2 = pd.DataFrame(res2)
    res2.to_csv('./result2/RI.csv', index=False, header=None, sep=',')
    TT = time.time()
    print('final finished', TT - T0)






