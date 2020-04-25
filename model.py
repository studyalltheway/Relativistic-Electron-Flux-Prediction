import tensorflow as tf
import numpy as np
import gc
gc.disable()

## Feedforward neural network with a scaling transformation layer
def create_model1(x, y0, num_in, reg1=0.001, reg2=0.001, node1=20, node2=20, seed=1, weight_init=1, Loss='L2', f1='selu', f2='sigmoid'):
    w1 = tf.Variable(tf.truncated_normal([num_in, node1], stddev=0.01, seed=seed+1), name='w1')
    w2 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=seed+11), name='w2')
    w3 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=seed+21), name='w3')
    b1 = tf.Variable(tf.zeros([node1]), name='b1')
    b2 = tf.Variable(tf.zeros([node2]), name='b2')
    b3 = tf.Variable(tf.zeros([1]), name='b3')
    inputweight = tf.Variable(tf.zeros([num_in]) + weight_init, name='weight')

    x1 = tf.multiply(x, inputweight) + tf.Variable(tf.zeros([num_in]))
    a = tf.matmul(x1, w1) + b1
    if f1 == 'selu':
        a = tf.nn.selu(a)
    elif f1[0:2] == 'le':
        alpha = float(f1.split('_')[-1])
        a = tf.nn.leaky_relu(a, alpha=alpha)
    elif f1 == 'relu':
        a = tf.nn.relu(a)
    elif f1 == 'sigmoid':
        a = tf.nn.sigmoid(a)
    elif f1 == 'tanh':
        a = tf.nn.tanh(a)
    else:
        raise NameError

    a2 = tf.matmul(a, w2) + b2
    if f2 == 'selu':
        a2 = tf.nn.selu(a2)
    elif f2[0:2] == 'le':
        alpha = float(f2.split('_')[-1])
        a2 = tf.nn.leaky_relu(a2, alpha=alpha)
    elif f2 == 'relu':
        a2 = tf.nn.relu(a2)
    elif f2 == 'sigmoid':
        a2 = tf.nn.sigmoid(a2)
    elif f2 == 'tanh':
        a2 = tf.nn.tanh(a2)
    else:
        raise NameError
    y1 = tf.matmul(a2, w3) + b3

    reg1 = reg1
    reg2 = reg2
    tf.add_to_collection('Loss2', tf.contrib.layers.l1_regularizer(reg1)(inputweight))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w3))

    if Loss == 'L2':
        loss = tf.reduce_mean(tf.square(y1 - y0))
    elif Loss == 'L1':
        loss = tf.reduce_mean(tf.abs(y1 - y0))
    elif Loss[0] == 't':
        tau = float(Loss[1:])
        loss = tf.reduce_mean(tf.where(tf.greater(y1, y0), (y1-y0)*tau, (y0-y1)*(1-tau)))
    elif Loss[0] == 'm':
        threshold = float(Loss[1:])
        loss = tf.reduce_mean(tf.where(tf.greater(y1, threshold), (y1-y0)**2, abs(y0-y1)))
    else:
        raise NameError
    totalloss = loss + tf.add_n(tf.get_collection('Loss2'))
    return y1, totalloss, loss, inputweight


### MLP series
def create_model2(x, y0, num_in, reg2=0.001, node1=20, node2=20, seed=1, Loss='L2', f1='selu', f2='sigmoid'):
    w1 = tf.Variable(tf.truncated_normal([num_in, node1], stddev=0.01, seed=seed+1))
    w2 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=seed+11))
    w3 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=seed+21))
    b1 = tf.Variable(tf.zeros([node1]))
    b2 = tf.Variable(tf.zeros([node2]))
    b3 = tf.Variable(tf.zeros([1]))

    a = tf.matmul(x, w1) + b1
    if f1 == 'selu':
        a = tf.nn.selu(a)
    elif f1[0:2] == 'le':
        alpha = float(f1.split('_')[-1])
        a = tf.nn.leaky_relu(a, alpha=alpha)
    elif f1 == 'relu':
        a = tf.nn.relu(a)
    elif f1 == 'sigmoid':
        a = tf.nn.sigmoid(a)
    elif f2 == 'tanh':
        a = tf.nn.tanh(a)
    else:
        raise NameError

    a2 = tf.matmul(a, w2) + b2
    if f2 == 'selu':
        a2 = tf.nn.selu(a2)
    elif f2[0:2] == 'le':
        alpha = float(f2.split('_')[-1])
        a2 = tf.nn.leaky_relu(a2, alpha=alpha)
    elif f2 == 'relu':
        a2 = tf.nn.relu(a2)
    elif f2 == 'sigmoid':
        a2 = tf.nn.sigmoid(a2)
    elif f2 == 'tanh':
        a2 = tf.nn.tanh(a2)
    else:
        raise NameError
    y1 = tf.matmul(a2, w3) + b3

    reg2 = reg2
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w3))
    if Loss == 'L2':
        loss = tf.reduce_mean(tf.square(y1 - y0))
    elif Loss == 'L1':
        loss = tf.reduce_mean(tf.abs(y1 - y0))
    elif Loss[0] == 't':
        tau = float(Loss[1:])
        loss = tf.reduce_mean(tf.where(tf.greater(y1, y0), (y1-y0)*tau, (y0-y1)*(1-tau)))
    elif Loss[0] == 'm':
        threshold = float(Loss[1:])
        loss = tf.reduce_mean(tf.where(tf.greater(y1, threshold), (y1-y0)**2, abs(y0-y1)))
    else:
        raise NameError
    totalloss = loss + tf.add_n(tf.get_collection('Loss2'))
    return y1, totalloss, loss

### MLP series —— n(1, 2, 3) hidden layer MLP
def create_model2_1(x, y0, num_in, layer=2, reg2=0.001, node1=20, node2=30, node3=20, seed=1, Loss='L2', f1='sigmoid', f2='sigmoid', f3='sigmoid'):
    w1 = tf.Variable(tf.truncated_normal([num_in, node1], stddev=0.01, seed=seed+1))
    b1 = tf.Variable(tf.zeros([node1]))
    a = tf.matmul(x, w1) + b1
    if f1 == 'selu':
        a = tf.nn.selu(a)
    elif f1[0:2] == 'le':
        alpha = float(f1.split('_')[-1])
        a = tf.nn.leaky_relu(a, alpha=alpha)
    elif f1 == 'relu':
        a = tf.nn.relu(a)
    elif f1 == 'sigmoid':
        a = tf.nn.sigmoid(a)
    elif f1 == 'tanh':
        a = tf.nn.tanh(a)
    else:
        raise NameError

    if layer == 1:
        w2 = tf.Variable(tf.truncated_normal([node1, 1], stddev=0.01, seed=seed+11))
        b2 = tf.Variable(tf.zeros([1]))
        y1 = tf.matmul(a, w2) + b2
    elif layer == 2:
        w2 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=seed+11))
        w3 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=seed+21))
        b2 = tf.Variable(tf.zeros([node2]))
        b3 = tf.Variable(tf.zeros([1]))
        a2 = tf.matmul(a, w2) + b2
        if f2 == 'sigmoid':
            a2 = tf.nn.sigmoid(a2)
        elif f2 == 'relu':
            a2 = tf.nn.relu(a2)
        elif f2 == 'sigmoid':
            a2 = tf.nn.sigmoid(a2)
        elif f2 == 'tanh':
            a2 = tf.nn.tanh(a2)
        else:
            raise NameError
        y1 = tf.matmul(a2, w3) + b3
    elif layer == 3:
        w2 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=seed + 11))
        w3 = tf.Variable(tf.truncated_normal([node2, node3], stddev=0.01, seed=seed + 21))
        w4 = tf.Variable(tf.truncated_normal([node3, 1], stddev=0.01, seed=seed + 31))
        b2 = tf.Variable(tf.zeros([node2]))
        b3 = tf.Variable(tf.zeros([node3]))
        b4 = tf.Variable(tf.zeros([1]))
        a2 = tf.matmul(a, w2) + b2
        if f2 == 'sigmoid':
            a2 = tf.nn.sigmoid(a2)
        elif f2 == 'relu':
            a2 = tf.nn.relu(a2)
        elif f2 == 'sigmoid':
            a2 = tf.nn.sigmoid(a2)
        elif f2 == 'tanh':
            a2 = tf.nn.tanh(a2)
        else:
            raise NameError
        a3 = tf.matmul(a2, w3) + b3
        if f3 == 'sigmoid':
            a3 = tf.nn.sigmoid(a3)
        elif f3 == 'relu':
            a3 = tf.nn.relu(a3)
        elif f3 == 'sigmoid':
            a3 = tf.nn.sigmoid(a3)
        elif f3 == 'tanh':
            a3 = tf.nn.tanh(a3)
        else:
            raise NameError
        y1 = tf.matmul(a3, w4) + b4

    reg2 = reg2
    if layer == 1:
        tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1))
    elif layer == 2:
        tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1))
        tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2))
    elif layer == 3:
        tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1))
        tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2))
        tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w3))
    if Loss == 'L2':
        loss = tf.reduce_mean(tf.square(y1 - y0))
    elif Loss == 'L1':
        loss = tf.reduce_mean(tf.abs(y1 - y0))
    elif Loss[0] == 't':
        tau = float(Loss[1:])
        loss = tf.reduce_mean(tf.where(tf.greater(y1, y0), (y1-y0)*tau, (y0-y1)*(1-tau)))
    elif Loss[0] == 'm':
        threshold = float(Loss[1:])
        loss = tf.reduce_mean(tf.where(tf.greater(y1, threshold), (y1-y0)**2, abs(y0-y1)))
    else:
        raise NameError
    totalloss = loss + tf.add_n(tf.get_collection('Loss2'))
    return y1, totalloss, loss


## model for Grad_FFN;  check the gradient information
def create_model3(x, y0, num_in, reg1=0.001, reg2=0.001, node1=20, node2=20, seed=1, weight_init=1):
    w1 = tf.Variable(tf.truncated_normal([num_in, node1], stddev=0.01, seed=seed+1), name='w1')
    w2 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=seed+11), name='w2')
    w3 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=seed+21), name='w3')
    b1 = tf.Variable(tf.zeros([node1]), name='b1')
    b2 = tf.Variable(tf.zeros([node2]), name='b2')
    b3 = tf.Variable(tf.zeros([1]), name='b3')
    inputweight = tf.Variable(tf.zeros([num_in]) + weight_init, name='weight')

    x1 = tf.multiply(x, inputweight) + tf.Variable(tf.zeros([num_in]))
    a = tf.matmul(x1, w1) + b1
    a = tf.nn.selu(a)
    a2 = tf.matmul(a, w2) + b2
    a2 = tf.nn.sigmoid(a2)
    y1 = tf.matmul(a2, w3) + b3
    reg1 = reg1
    reg2 = reg2
    tf.add_to_collection('Loss2', tf.contrib.layers.l1_regularizer(reg1)(inputweight))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w3))
    loss = tf.reduce_mean(tf.square(y1 - y0))
    totalloss = loss + tf.add_n(tf.get_collection('Loss2'))
    return y1, totalloss, loss, inputweight, x1, a, a2


## model of Quan_reg1, quantile regression
def create_model4(x, y0, num_in, tau, reg1=0.001, reg2=0.001, node1=20, node2=20, seed=1, weight_init=1):
    w1 = tf.Variable(tf.truncated_normal([num_in, node1], stddev=0.01, seed=seed+1), name='w1')
    w2 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=seed+11), name='w2')
    w3 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=seed+21), name='w3')
    b1 = tf.Variable(tf.zeros([node1]), name='b1')
    b2 = tf.Variable(tf.zeros([node2]), name='b2')
    b3 = tf.Variable(tf.zeros([1]), name='b3')
    inputweight = tf.Variable(tf.zeros([num_in]) + weight_init, name='weight')

    x1 = tf.multiply(x, inputweight) + tf.Variable(tf.zeros([num_in]))
    a = tf.matmul(x1, w1) + b1
    a = tf.nn.selu(a)
    a2 = tf.matmul(a, w2) + b2
    a2 = tf.nn.sigmoid(a2)
    y1 = tf.matmul(a2, w3) + b3
    reg1 = reg1
    reg2 = reg2
    tf.add_to_collection('Loss2', tf.contrib.layers.l1_regularizer(reg1)(inputweight))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w3))
    loss = tf.reduce_mean(tf.where(tf.greater(y1, y0), (y1-y0)*tau, (y0-y1)*(1-tau)))
    totalloss = loss + tf.add_n(tf.get_collection('Loss2'))
    return y1, totalloss, loss, inputweight


## model of Quan_reg2, quantile regression for predicting 0.1-0.9 9 quantiles at the same time
def create_model4_2(x, y0, num_in, tau=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], pterm=10, \
                    reg1=0.001, reg2=0.001, node1=30, node2=30, weight_init=1):
    # the main part of the network
    w0 = tf.Variable(tf.truncated_normal([num_in, node1], stddev=0.01, seed=1), name='w0')
    b0 = tf.Variable(tf.zeros([node1]), name='b0')
    inputweight = tf.Variable(tf.zeros([num_in]) + weight_init, name='weight')
    x1 = tf.multiply(x, inputweight) + tf.Variable(tf.zeros([num_in]))
    a = tf.matmul(x1, w0) + b0
    a = tf.nn.selu(a)
    # tau = 0.1
    w1 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=1), name='w1')
    b1 = tf.Variable(tf.zeros([node2]), name='b1')
    w1_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=1), name='w1_2')
    b1_2 = tf.Variable(tf.zeros([1]), name='b1_2')
    a1 = tf.matmul(a, w1) + b1
    a1 = tf.nn.sigmoid(a1)
    y1 = tf.matmul(a1, w1_2) + b1_2
    # tau = 0.2
    w2 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=1), name='w2')
    b2 = tf.Variable(tf.zeros([node2]), name='b2')
    w2_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=1), name='w2_2')
    b2_2 = tf.Variable(tf.zeros([1]), name='b2_2')
    a2 = tf.matmul(a, w2) + b2
    a2 = tf.nn.sigmoid(a2)
    y2 = tf.matmul(a2, w2_2) + b2_2
    # tau = 0.3
    w3 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=1), name='w3')
    b3 = tf.Variable(tf.zeros([node2]), name='b3')
    w3_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=1), name='w3_2')
    b3_2 = tf.Variable(tf.zeros([1]), name='b3_2')
    a3 = tf.matmul(a, w3) + b3
    a3 = tf.nn.sigmoid(a3)
    y3 = tf.matmul(a3, w3_2) + b3_2
    # tau = 0.4
    w4 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=1), name='w4')
    b4 = tf.Variable(tf.zeros([node2]), name='b4')
    w4_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=1), name='w4_2')
    b4_2 = tf.Variable(tf.zeros([1]), name='b4_2')
    a4 = tf.matmul(a, w4) + b4
    a4 = tf.nn.sigmoid(a4)
    y4 = tf.matmul(a4, w4_2) + b4_2
    # tau = 0.5
    w5 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=1), name='w5')
    b5 = tf.Variable(tf.zeros([node2]), name='b5')
    w5_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=1), name='w5_2')
    b5_2 = tf.Variable(tf.zeros([1]), name='b5_2')
    a5 = tf.matmul(a, w5) + b5
    a5 = tf.nn.sigmoid(a5)
    y5 = tf.matmul(a5, w5_2) + b5_2
    # tau = 0.6
    w6 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=1), name='w6')
    b6 = tf.Variable(tf.zeros([node2]), name='b6')
    w6_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=1), name='w6_2')
    b6_2 = tf.Variable(tf.zeros([1]), name='b6_2')
    a6 = tf.matmul(a, w6) + b6
    a6 = tf.nn.sigmoid(a6)
    y6 = tf.matmul(a6, w6_2) + b6_2
    # tau = 0.7
    w7 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=1), name='w7')
    b7 = tf.Variable(tf.zeros([node2]), name='b7')
    w7_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=1), name='w7_2')
    b7_2 = tf.Variable(tf.zeros([1]), name='b7_2')
    a7 = tf.matmul(a, w7) + b7
    a7 = tf.nn.sigmoid(a7)
    y7 = tf.matmul(a7, w7_2) + b7_2
    # tau = 0.8
    w8 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=1), name='w8')
    b8 = tf.Variable(tf.zeros([node2]), name='b8')
    w8_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=1), name='w8_2')
    b8_2 = tf.Variable(tf.zeros([1]), name='b8_2')
    a8 = tf.matmul(a, w8) + b8
    a8 = tf.nn.sigmoid(a8)
    y8 = tf.matmul(a8, w8_2) + b8_2
    # tau = 0.8
    w9 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=1), name='w9')
    b9 = tf.Variable(tf.zeros([node2]), name='b9')
    w9_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=1), name='w9_2')
    b9_2 = tf.Variable(tf.zeros([1]), name='b9_2')
    a9 = tf.matmul(a, w9) + b9
    a9 = tf.nn.sigmoid(a9)
    y9 = tf.matmul(a9, w9_2) + b9_2

    reg1 = reg1
    reg2 = reg2
    tf.add_to_collection('Loss2', tf.contrib.layers.l1_regularizer(reg1)(inputweight))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w0))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w3))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w3_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w4))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w4_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w5))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w5_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w6))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w6_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w7))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w7_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w8))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w8_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w9))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w9_2))

    tau = tau
    pterm = pterm
    loss1 = tf.reduce_mean(tf.where(tf.greater(y1, y0), (y1-y0)*tau[0], (y0-y1)*(1-tau[0])))
    loss2 = tf.reduce_mean(tf.where(tf.greater(y2, y0), (y2-y0)*tau[1], (y0-y2)*(1-tau[1])))
    loss3 = tf.reduce_mean(tf.where(tf.greater(y3, y0), (y3-y0)*tau[2], (y0-y3)*(1-tau[2])))
    loss4 = tf.reduce_mean(tf.where(tf.greater(y4, y0), (y4-y0)*tau[3], (y0-y4)*(1-tau[3])))
    loss5 = tf.reduce_mean(tf.where(tf.greater(y5, y0), (y5-y0)*tau[4], (y0-y5)*(1-tau[4])))
    loss6 = tf.reduce_mean(tf.where(tf.greater(y6, y0), (y6-y0)*tau[4], (y0-y6)*(1-tau[5])))
    loss7 = tf.reduce_mean(tf.where(tf.greater(y7, y0), (y7-y0)*tau[4], (y0-y7)*(1-tau[6])))
    loss8 = tf.reduce_mean(tf.where(tf.greater(y8, y0), (y8-y0)*tau[4], (y0-y8)*(1-tau[7])))
    loss9 = tf.reduce_mean(tf.where(tf.greater(y9, y0), (y9-y0)*tau[4], (y0-y9)*(1-tau[8])))
    p1 = tf.reduce_mean(tf.where(tf.greater(y1, y2), y2*0, (y2-y1)*pterm))
    p2 = tf.reduce_mean(tf.where(tf.greater(y2, y3), y3*0, (y3-y2)*pterm))
    p3 = tf.reduce_mean(tf.where(tf.greater(y3, y4), y4*0, (y4-y3)*pterm))
    p4 = tf.reduce_mean(tf.where(tf.greater(y4, y5), y5*0, (y5-y4)*pterm))
    p5 = tf.reduce_mean(tf.where(tf.greater(y5, y6), y6*0, (y6-y5)*pterm))
    p6 = tf.reduce_mean(tf.where(tf.greater(y6, y7), y7*0, (y7-y6)*pterm))
    p7 = tf.reduce_mean(tf.where(tf.greater(y7, y8), y8*0, (y8-y7)*pterm))
    p8 = tf.reduce_mean(tf.where(tf.greater(y8, y9), y9*0, (y9-y8)*pterm))

    loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
    totalloss = loss+p1+p2+p3+p4+p5+p6+p7+p8+tf.add_n(tf.get_collection('Loss2'))
    return [y1, y2, y3, y4, y5, y6, y7, y8, y9, loss, totalloss]


## model of Quan_reg2, three quantiles prediction at the same time [0.2, 0.5, 0.8] tau_range
def create_model4_3(x, y0, num_in, tau=[0.2, 0.5, 0.8], pterm=1, \
                    reg1=0.001, reg2=0.001, node1=25, node2=25, seed=1, weight_init=1):
    # main part of the network
    w0 = tf.Variable(tf.truncated_normal([num_in, node1], stddev=0.01, seed=seed+1), name='w0')
    b0 = tf.Variable(tf.zeros([node1]), name='b0')
    inputweight = tf.Variable(tf.zeros([num_in]) + weight_init, name='weight')
    x1 = tf.multiply(x, inputweight) + tf.Variable(tf.zeros([num_in]))
    a = tf.matmul(x1, w0) + b0
    a = tf.nn.selu(a)
    # tau = 0.2
    w1 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=seed+11), name='w1')
    b1 = tf.Variable(tf.zeros([node2]), name='b1')
    w1_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=seed+21), name='w1_2')
    b1_2 = tf.Variable(tf.zeros([1]), name='b1_2')
    a1 = tf.matmul(a, w1) + b1
    a1 = tf.nn.sigmoid(a1)
    y1 = tf.matmul(a1, w1_2) + b1_2
    # tau = 0.5
    w2 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=seed+31), name='w2')
    b2 = tf.Variable(tf.zeros([node2]), name='b2')
    w2_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=seed+41), name='w2_2')
    b2_2 = tf.Variable(tf.zeros([1]), name='b2_2')
    a2 = tf.matmul(a, w2) + b2
    a2 = tf.nn.sigmoid(a2)
    y2 = tf.matmul(a2, w2_2) + b2_2
    # tau = 0.8
    w3 = tf.Variable(tf.truncated_normal([node1, node2], stddev=0.01, seed=seed+51), name='w3')
    b3 = tf.Variable(tf.zeros([node2]), name='b3')
    w3_2 = tf.Variable(tf.truncated_normal([node2, 1], stddev=0.01, seed=seed+61), name='w3_2')
    b3_2 = tf.Variable(tf.zeros([1]), name='b3_2')
    a3 = tf.matmul(a, w3) + b3
    a3 = tf.nn.sigmoid(a3)
    y3 = tf.matmul(a3, w3_2) + b3_2

    reg1 = reg1
    reg2 = reg2
    tf.add_to_collection('Loss2', tf.contrib.layers.l1_regularizer(reg1)(inputweight))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w0))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w1_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w2_2))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w3))
    tf.add_to_collection('Loss2', tf.contrib.layers.l2_regularizer(reg2)(w3_2))

    tau = tau
    pterm = pterm
    loss1 = tf.reduce_mean(tf.where(tf.greater(y1, y0), (y1-y0)*tau[0], (y0-y1)*(1-tau[0])))
    loss2 = tf.reduce_mean(tf.where(tf.greater(y2, y0), (y2-y0)*tau[1], (y0-y2)*(1-tau[1])))
    loss3 = tf.reduce_mean(tf.where(tf.greater(y3, y0), (y3-y0)*tau[2], (y0-y3)*(1-tau[2])))
    p1 = tf.reduce_mean(tf.where(tf.greater(y1, y2), y2*0, (y2-y1)*pterm))
    p2 = tf.reduce_mean(tf.where(tf.greater(y2, y3), y3*0, (y3-y2)*pterm))

    loss = loss1+loss2+loss3
    totalloss = loss+p1+p2+tf.add_n(tf.get_collection('Loss2'))
    return [y1, y2, y3, loss, totalloss]
