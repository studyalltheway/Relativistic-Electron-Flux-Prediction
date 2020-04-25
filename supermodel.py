import numpy as np
import tensorflow as tf
from utils import RMSE, PE, MSE, corrcal, LossQ
from model import create_model1, create_model2, create_model4, create_model4_2, create_model4_3
import gc
gc.disable()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

### feedforward neural network (10 cross validation)
def supermodel1(Set, Label, ShowSet, ShowLabel, config):
    import time
    T0 = time.time()
    
    trloss, teloss, shloss, trainPE, testPE, showPE, trainCC, testCC, showCC, pre1, pre2, pre3, weight = \
        [], [], [], [], [], [], [], [], [], [], [], [], []

    seed = 1
    ## parameters of the network
    num_in = 85
    node1 = 25
    node2 = 25
    weight_init = 1
    ## parameters of the training process
    batch_size=200
    epochs = 1500
    learnrate = 0.01
    reg1 = 0.001
    reg2 = 0.001
    Loss = 'L2'
    f1 = 'selu'
    f2 = 'sigmoid'

    if hasattr(config, 'seed'):
        seed = config.seed

    if hasattr(config, 'num_in'):
        num_in = config.num_in
    if hasattr(config, 'node1'):
        node1 = config.node1
    if hasattr(config, 'node2'):
        node2 = config.node2
    if hasattr(config, 'weight_init'):
        weight_init = config.weight_init

    if hasattr(config, 'batch_size'):
        batch_size = config.batch_size
    if hasattr(config, 'epochs'):
        epochs = config.epochs
    if hasattr(config, 'learnrate'):
        learnrate = config.learnrate
    if hasattr(config, 'reg1'):
        reg1 = config.reg1
    if hasattr(config, 'reg2'):
        reg2 = config.reg2
    if hasattr(config, 'Loss'):
        Loss = config.Loss
    if hasattr(config, 'fun1'):
        f1 = config.fun1
    if hasattr(config, 'fun2'):
        f2 = config.fun2

    ## model training
    splitnum = len(Set)//10
    NN = Set.shape[1]
    for i in range(10):
        # make data sets
        srange=np.arange(splitnum*i, splitnum*i+splitnum)
        testset=Set[srange, :]
        testlabel=Label[srange, :]
        trainset=np.delete(Set, srange, axis=0)
        trainlabel=np.delete(Label, srange, axis=0)

        # training set shuffle
        data2 = np.hstack((trainset, trainlabel))
        np.random.seed(2020)  # This year is 2020
        np.random.shuffle(data2)
        trainset = data2[:, 0:-1]
        trainlabel = data2[:, -1].reshape(-1, 1)

        ## build the model
        x = tf.placeholder(tf.float32, [None, num_in])
        y0 = tf.placeholder(tf.float32, [None, 1])
        y1, totalloss, loss, inputweight = create_model1( \
            x, y0, num_in, reg1=reg1, reg2=reg2, node1=node1, node2=node2, seed=seed, \
            weight_init=weight_init, Loss=Loss, f1=f1, f2=f2)

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
                # if ii%200==0:
                #     print('Loss:', sess.run(loss, feed_dict={x: X[start:end], y0: Y[start:end]}))
                #     print('Total Loss:', sess.run(totalloss, feed_dict={x: X[start:end], y0: Y[start:end]}))
            trainloss = sess.run(loss, feed_dict={x: X, y0: Y})
            print('##-------------------#', i)
            print('total_loss1:', trainloss)
            ###  evaluation of the results
            prediction1 = sess.run(y1, feed_dict={x: trainset})
            prediction2 = sess.run(y1, feed_dict={x: testset})
            prediction3 = sess.run(y1, feed_dict={x: ShowSet})
            pre2.append(prediction2.reshape(-1))
            pre3.append(prediction3.reshape(-1))

            # true value of trainlabel and prediction1
            trainlabel += 8
            prediction1 += 8
            trloss.append(RMSE(trainlabel, prediction1))
            trainPE.append(PE(trainlabel, prediction1))
            trainCC.append(corrcal(trainlabel, prediction1))
            weight.append(sess.run(inputweight))

    prediction2 = np.array(pre2).reshape(-1)
    prediction3 = np.mean(np.array(pre3), axis=0)
    # true value of prediction2 and prediction3, Label and ShowLabel
    prediction2 += 8
    prediction3 += 8
    Label += 8
    ShowLabel += 8

    trloss = np.mean(np.array(trloss))
    trainPE = np.mean(np.array(trainPE))
    trainCC = np.mean(np.array(trainCC))

    # calculate teloss, shloss, testPE, showPE
    ## attention, due to "//10", the datasize may not be consistent
    prediction2_num = len(prediction2)
    teloss = RMSE(Label.reshape(-1)[0:prediction2_num], prediction2)
    testPE = PE(Label.reshape(-1)[0:prediction2_num], prediction2)
    testCC = corrcal(Label.reshape(-1)[0:prediction2_num], prediction2)

    shloss = RMSE(ShowLabel.reshape(-1), prediction3)
    showPE = PE(ShowLabel.reshape(-1), prediction3)
    showCC = corrcal(ShowLabel.reshape(-1), prediction3)

    weight = np.mean(np.array(weight), axis=0)
    #weightvar = np.var(weight)
    #print('weight var:', weightvar)
    TT = time.time()
    print('Time', TT-T0)
    print('>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    return trloss, teloss, shloss, trainPE, testPE, showPE, trainCC, testCC, showCC, TT-T0, weight


### feedforward neural network (10 cross validation)
# the same as supermodel1, but save the predicted flux into files
# By adding 8, the true values can be gotten
def supermodel1_2(Set, Label, ShowSet, ShowLabel, config):

    pre1, pre2, pre3 = [], [], []

    seed = 1
    ## parameters of the network
    num_in = 100
    node1 = 25
    node2 = 25
    weight_init = 1
    ## parameters of the training process
    batch_size=200
    epochs = 1500
    learnrate = 0.01
    reg1 = 0.001
    reg2 = 0.001
    Loss = 'L2'
    f1 = 'selu'
    f2 = 'sigmoid'

    if hasattr(config, 'seed'):
        seed = config.seed

    if hasattr(config, 'num_in'):
        num_in = config.num_in
    if hasattr(config, 'node1'):
        node1 = config.node1
    if hasattr(config, 'node2'):
        node2 = config.node2
    if hasattr(config, 'weight_init'):
        weight_init = config.weight_init

    if hasattr(config, 'batch_size'):
        batch_size = config.batch_size
    if hasattr(config, 'epochs'):
        epochs = config.epochs
    if hasattr(config, 'learnrate'):
        learnrate = config.learnrate
    if hasattr(config, 'reg1'):
        reg1 = config.reg1
    if hasattr(config, 'reg2'):
        reg2 = config.reg2
    if hasattr(config, 'Loss'):
        Loss = config.Loss
    if hasattr(config, 'fun1'):
        f1 = config.fun1
    if hasattr(config, 'fun2'):
        f2 = config.fun2

    ## model training
    splitnum = len(Set)//10
    NN = Set.shape[1]
    for i in range(10):
        # make data sets
        srange=np.arange(splitnum*i, splitnum*i+splitnum)
        testset=Set[srange, :]
        testlabel=Label[srange, :]
        trainset=np.delete(Set, srange, axis=0)
        trainlabel=np.delete(Label, srange, axis=0)

        # training set shuffle
        data2 = np.hstack((trainset, trainlabel))
        np.random.seed(2020)  # This year is 2020
        np.random.shuffle(data2)
        trainset = data2[:, 0:-1]
        trainlabel = data2[:, -1].reshape(-1, 1)

        ## build the model
        x = tf.placeholder(tf.float32, [None, num_in])
        y0 = tf.placeholder(tf.float32, [None, 1])
        y1, totalloss, loss, inputweight = create_model1( \
            x, y0, num_in, reg1=reg1, reg2=reg2, node1=node1, node2=node2, seed=seed, \
            weight_init=weight_init, Loss=Loss, f1=f1, f2=f2)

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
            trainloss = sess.run(loss, feed_dict={x: X, y0: Y})
            print('##-------------------#', i)
            print('total_loss1:', trainloss)
            ###  evaluation of the results
            prediction2 = sess.run(y1, feed_dict={x: testset})
            prediction3 = sess.run(y1, feed_dict={x: ShowSet})
            pre2.append(prediction2.reshape(-1))
            pre3.append(prediction3.reshape(-1))


    prediction2 = np.array(pre2).reshape(-1)
    prediction3 = np.mean(np.array(pre3), axis=0)

    print('>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    return prediction3


### MLP series
def supermodel2(Set, Label, ShowSet, ShowLabel, config):
    import time
    T0 = time.time()

    trloss, teloss, shloss, trainPE, testPE, showPE, trainCC, testCC, showCC, pre1, pre2, pre3 = \
        [], [], [], [], [], [], [], [], [], [], [], []

    seed = 1
    ## parameters of the network
    num_in = 85
    node1 = 25
    node2 = 25
    ## parameters of training
    batch_size = 200
    epochs = 1500
    learnrate = 0.01
    reg2 = 0.001
    Loss = 'L2'
    f1 = 'selu'
    f2 = 'sigmoid'

    if hasattr(config, 'seed'):
        seed = config.seed

    if hasattr(config, 'num_in'):
        num_in = config.num_in
    if hasattr(config, 'node1'):
        node1 = config.node1
    if hasattr(config, 'node2'):
        node2 = config.node2

    if hasattr(config, 'batch_size'):
        batch_size = config.batch_size
    if hasattr(config, 'epochs'):
        epochs = config.epochs
    if hasattr(config, 'learnrate'):
        learnrate = config.learnrate
    if hasattr(config, 'reg2'):
        reg2 = config.reg2
    if hasattr(config, 'Loss'):
        Loss = config.Loss
    if hasattr(config, 'fun1'):
        f1 = config.fun1
    if hasattr(config, 'fun2'):
        f2 = config.fun2

    ## model training (10 cross validation)
    splitnum = len(Set)//10
    NN = Set.shape[1]
    for i in range(10):
        # make data sets
        srange=np.arange(splitnum*i, splitnum*i+splitnum)
        testset=Set[srange, :]
        testlabel=Label[srange, :]
        trainset=np.delete(Set, srange, axis=0)
        trainlabel=np.delete(Label, srange, axis=0)

        # training set shuffle
        data2 = np.hstack((trainset, trainlabel))
        np.random.seed(2020)  # This year is 2020
        np.random.shuffle(data2)
        trainset = data2[:, 0:-1]
        trainlabel = data2[:, -1].reshape(-1, 1)

        ## build model
        x = tf.placeholder(tf.float32, [None, num_in])
        y0 = tf.placeholder(tf.float32, [None, 1])
        y1, totalloss, loss = create_model2( \
            x, y0, num_in, reg2=reg2, node1=node1, node2=node2, \
            seed=seed, Loss=Loss, f1=f1, f2=f2)

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
                # if ii%200==0:
                #     print('Loss:', sess.run(loss, feed_dict={x: X[start:end], y0: Y[start:end]}))
                #     print('Total Loss:', sess.run(totalloss, feed_dict={x: X[start:end], y0: Y[start:end]}))
            trainloss = sess.run(loss, feed_dict={x: X, y0: Y})
            print('##-------------------#', i)
            print('total_loss1:', trainloss)
            ### evaluation of the results
            prediction1 = sess.run(y1, feed_dict={x: trainset})
            prediction2 = sess.run(y1, feed_dict={x: testset})
            prediction3 = sess.run(y1, feed_dict={x: ShowSet})
            pre2.append(prediction2.reshape(-1))
            pre3.append(prediction3.reshape(-1))
            # true value of trainlabel and prediction1
            trainlabel += 8
            prediction1 += 8

            trloss.append(RMSE(trainlabel, prediction1))
            trainPE.append(PE(trainlabel, prediction1))
            trainCC.append(corrcal(trainlabel, prediction1))

    prediction2 = np.array(pre2).reshape(-1)
    prediction3 = np.mean(np.array(pre3), axis=0)
    # true value of prediction2 and prediction3, Label and ShowLabel
    prediction2 += 8
    prediction3 += 8
    Label += 8
    ShowLabel += 8

    trloss = np.mean(np.array(trloss))
    trainPE = np.mean(np.array(trainPE))
    trainCC = np.mean(np.array(trainCC))

    # calculate teloss, shloss, testPE, showPE
    ## attention, due to "//10", the datasize may not be consistent
    prediction2_num = len(prediction2)
    teloss = RMSE(Label.reshape(-1)[0:prediction2_num], prediction2)
    testPE = PE(Label.reshape(-1)[0:prediction2_num], prediction2)
    testCC = corrcal(Label.reshape(-1)[0:prediction2_num], prediction2)

    shloss = RMSE(ShowLabel.reshape(-1), prediction3)
    showPE = PE(ShowLabel.reshape(-1), prediction3)
    showCC = corrcal(ShowLabel.reshape(-1), prediction3)

    TT = time.time()
    print('Time', TT-T0)
    print('>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    return trloss, teloss, shloss, trainPE, testPE, showPE, trainCC, testCC, showCC, TT-T0


# for Quan_reg1, quantile regression
def supermodel3(Set, Label, ShowSet, ShowLabel, config):
    import time
    T0 = time.time()

    trloss, teloss, shloss, pre1, pre2, pre3 = \
        [], [], [], [], [], []
    ##  parameters of network
    num_in = 85
    seed = 1
    node1 = 25
    node2 = 25
    weight_init = 1
    ##  parameters of training
    batch_size = 200
    epochs = 1500
    learnrate = 0.01
    reg1 = 0.001
    reg2 = 0.001
    ##  quantile regression: tau
    tau = 0.5

    if hasattr(config, 'tau'):
        tau = config.tau
    if hasattr(config, 'num_in'):
        num_in = config.num_in
    if hasattr(config, 'seed'):
        seed = config.seed
    if hasattr(config, 'node1'):
        node1 = config.node1
    if hasattr(config, 'node2'):
        node2 = config.node2
    if hasattr(config, 'weight_init'):
        weight_init = config.weight_init

    if hasattr(config, 'batch_size'):
        batch_size = config.batch_size
    if hasattr(config, 'epochs'):
        epochs = config.epochs
    if hasattr(config, 'learnrate'):
        learnrate = config.learnrate
    if hasattr(config, 'reg1'):
        reg1 = config.reg1
    if hasattr(config, 'reg2'):
        reg2 = config.reg2

    ## model training
    splitnum = len(Set)//10
    NN = Set.shape[1]
    for i in range(10):
        # make datasets
        srange=np.arange(splitnum*i, splitnum*i+splitnum)
        testset=Set[srange, :]
        testlabel=Label[srange, :]
        trainset=np.delete(Set, srange, axis=0)
        trainlabel=np.delete(Label, srange, axis=0)

        # training set shuffle
        data2 = np.hstack((trainset, trainlabel))
        np.random.seed(2020)  # This year is 2020
        np.random.shuffle(data2)
        trainset = data2[:, 0:-1]
        trainlabel = data2[:, -1].reshape(-1, 1)

        ### build model
        x = tf.placeholder(tf.float32, [None, num_in])
        y0 = tf.placeholder(tf.float32, [None, 1])
        y1, totalloss, loss, inputweight = create_model4( \
            x, y0, num_in, tau, reg1=reg1, reg2=reg2, node1=node1, node2=node2, seed=seed, weight_init=weight_init)

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
            #    if ii>1000 and ii%100==0:
             #       print('Loss:', sess.run(loss, feed_dict={x: X[start:end], y0: Y[start:end]}))
             #       print('Total Loss:', sess.run(totalloss, feed_dict={x: X[start:end], y0: Y[start:end]}))
            trainloss = sess.run(loss, feed_dict={x: X, y0: Y})
            print('##-------------------#', i)
            print('total_loss1:', trainloss)
            ## Evaluation
            prediction2 = sess.run(y1, feed_dict={x: testset})
            prediction3 = sess.run(y1, feed_dict={x: ShowSet})
            pre2.append(prediction2.reshape(-1))
            pre3.append(prediction3.reshape(-1))
            trloss.append(trainloss)


    prediction2 = np.array(pre2).reshape(-1)
    prediction3 = np.mean(np.array(pre3), axis=0)

    trloss = np.mean(np.array(trloss))
    ## attention, due to "//10", the datasize may not be consistent
    prediction2_num = len(prediction2)
    teloss = LossQ(Label.reshape(-1)[0:prediction2_num], prediction2, tau)
    shloss = LossQ(ShowLabel.reshape(-1), prediction3, tau)

    TT = time.time()
    print('Time', TT-T0)
    print('>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    return trloss, teloss, shloss, TT-T0, prediction2, prediction3


# for Quan_reg3, quantile regression for predicting many quantiles at the same time
def supermodel3_2(Set, Label, ShowSet, ShowLabel, config):
    import time
    T0 = time.time()

    pre1, pre2, pre3, pre4, pre5, pre6, pre7, pre8, pre9 = [], [], [], [], [], [], [], [], []
    trloss, teloss, shloss = [], [], []
    ## parameters of network
    num_in = 85
    node1 = 30
    node2 = 30
    weight_init = 1
    ## parameters of training
    batch_size = 300
    epochs = 2500
    learnrate = 0.01
    reg1 = 0.001
    reg2 = 0.001
    taurange = np.arange(0.1, 1, 0.1)
    pterm = 10

    if hasattr(config, 'num_in'):
        num_in = config.num_in
    if hasattr(config, 'node1'):
        node1 = config.node1
    if hasattr(config, 'node2'):
        node2 = config.node2

    if hasattr(config, 'batch_size'):
        batch_size = config.batch_size
    if hasattr(config, 'epochs'):
        epochs = config.epochs
    if hasattr(config, 'learnrate'):
        learnrate = config.learnrate
    if hasattr(config, 'reg1'):
        reg1 = config.reg1
    if hasattr(config, 'reg2'):
        reg2 = config.reg2
    if hasattr(config, 'pterm'):
        pterm = config.pterm

    ## model training
    splitnum = len(Set)//10
    NN = Set.shape[1]
    for i in range(10):
        # make data sets
        srange=np.arange(splitnum*i, splitnum*i+splitnum)
        testset=Set[srange, :]
        testlabel=Label[srange, :]
        trainset=np.delete(Set, srange, axis=0)
        trainlabel=np.delete(Label, srange, axis=0)

        # training set shuffle
        data2 = np.hstack((trainset, trainlabel))
        np.random.seed(2020)  # This year is 2020
        np.random.shuffle(data2)
        trainset = data2[:, 0:-1]
        trainlabel = data2[:, -1].reshape(-1, 1)

        ### build model
        x = tf.placeholder(tf.float32, [None, num_in])
        y0 = tf.placeholder(tf.float32, [None, 1])
        res = create_model4_2( \
            x, y0, num_in, tau=taurange, pterm=pterm, reg1=reg1, \
            reg2=reg2, node1=node1, node2=node2, weight_init=weight_init)
        # return [y1, y2, y3, y4, y5, loss, totalloss]

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
        y1, y2, y3, y4, y5, y6, y7, y8, y9 = res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]
        loss = res[9]
        totalloss = res[10]
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
                #     print('Show Loss:', sess.run(loss, feed_dict={x: ShowSet, y0: ShowLabel}))
            print('##-------------------#', i)
            trainloss = sess.run(loss, feed_dict={x: X, y0: Y})
            testloss = sess.run(loss, feed_dict={x: testset, y0: testlabel})
            showloss = sess.run(loss, feed_dict={x: ShowSet, y0: ShowLabel})
            trloss.append(trainloss)
            teloss.append(testloss)
            shloss.append(showloss)

            ## Evaluation
            p1 = sess.run(y1, feed_dict={x: ShowSet})
            p2 = sess.run(y2, feed_dict={x: ShowSet})
            p3 = sess.run(y3, feed_dict={x: ShowSet})
            p4 = sess.run(y4, feed_dict={x: ShowSet})
            p5 = sess.run(y5, feed_dict={x: ShowSet})
            p6 = sess.run(y6, feed_dict={x: ShowSet})
            p7 = sess.run(y7, feed_dict={x: ShowSet})
            p8 = sess.run(y8, feed_dict={x: ShowSet})
            p9 = sess.run(y9, feed_dict={x: ShowSet})
            pre1.append(p1.reshape(-1))
            pre2.append(p2.reshape(-1))
            pre3.append(p3.reshape(-1))
            pre4.append(p4.reshape(-1))
            pre5.append(p5.reshape(-1))
            pre6.append(p6.reshape(-1))
            pre7.append(p7.reshape(-1))
            pre8.append(p8.reshape(-1))
            pre9.append(p9.reshape(-1))

    trloss = np.mean(trloss)
    teloss = np.mean(teloss)
    shloss = np.mean(shloss)
    predict1 = np.mean(np.array(pre1), axis=0)
    predict2 = np.mean(np.array(pre2), axis=0)
    predict3 = np.mean(np.array(pre3), axis=0)
    predict4 = np.mean(np.array(pre4), axis=0)
    predict5 = np.mean(np.array(pre5), axis=0)
    predict6 = np.mean(np.array(pre6), axis=0)
    predict7 = np.mean(np.array(pre7), axis=0)
    predict8 = np.mean(np.array(pre8), axis=0)
    predict9 = np.mean(np.array(pre9), axis=0)

    TT = time.time()
    print('Time', TT-T0)
    print('>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    return [predict1, predict2, predict3, predict4, predict5, predict6, predict7, predict8, predict9, trloss, teloss, shloss]


# for Quan_reg2, quantile regression for predicting three quantiles at the same time
def supermodel3_3(Set, Label, ShowSet, ShowLabel, config):
    import time
    T0 = time.time()

    pre1, pre2, pre3 = [], [], []
    ##  parameters of network
    seed = 1
    num_in = 85
    node1 = 25
    node2 = 25
    weight_init = 1
    ## parameters of training
    batch_size = 300
    epochs = 1500
    learnrate = 0.01
    reg1 = 0.001
    reg2 = 0.001
    ## quantile regression: tau
    tau = 0.5
    taurange = [0.2, 0.5, 0.8]
    pterm = 1

    if hasattr(config, 'seed'):
        seed = config.seed
    if hasattr(config, 'num_in'):
        num_in = config.num_in
    if hasattr(config, 'node1'):
        node1 = config.node1
    if hasattr(config, 'node2'):
        node2 = config.node2
    if hasattr(config, 'weight_init'):
        weight_init = config.weight_init

    if hasattr(config, 'batch_size'):
        batch_size = config.batch_size
    if hasattr(config, 'epochs'):
        epochs = config.epochs
    if hasattr(config, 'learnrate'):
        learnrate = config.learnrate
    if hasattr(config, 'reg1'):
        reg1 = config.reg1
    if hasattr(config, 'reg2'):
        reg2 = config.reg2
    if hasattr(config, 'tau'):
        tau = config.tau
    if hasattr(config, 'taurange'):
        taurange = config.taurange
    if hasattr(config, 'pterm'):
        pterm = config.pterm

    ## model training
    trloss, teloss, shloss = [], [], []
    splitnum = len(Set)//10
    NN = Set.shape[1]
    for i in range(10):
        # make data sets
        srange=np.arange(splitnum*i, splitnum*i+splitnum)
        testset=Set[srange, :]
        testlabel=Label[srange, :]
        trainset=np.delete(Set, srange, axis=0)
        trainlabel=np.delete(Label, srange, axis=0)

        # training set shuffle
        data2 = np.hstack((trainset, trainlabel))
        np.random.seed(2020)  # This year is 2020
        np.random.shuffle(data2)
        trainset = data2[:, 0:-1]
        trainlabel = data2[:, -1].reshape(-1, 1)

        ### build model
        x = tf.placeholder(tf.float32, [None, num_in])
        y0 = tf.placeholder(tf.float32, [None, 1])
        res = create_model4_3( \
            x, y0, num_in, tau=taurange, pterm=pterm, reg1=reg1, \
            reg2=reg2, node1=node1, node2=node2, weight_init=weight_init)
        # return [y1, y2, y3, y4, y5, loss, totalloss]

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
        y1, y2, y3 = res[0], res[1], res[2]
        loss = res[3]
        totalloss = res[4]
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
             #   if ii%100==0:
             #       print('Loss:', sess.run(loss, feed_dict={x: X[start:end], y0: Y[start:end]}))
             #       print('Total Loss:', sess.run(totalloss, feed_dict={x: X[start:end], y0: Y[start:end]}))
             #       print('Show Loss:', sess.run(loss, feed_dict={x: ShowSet, y0: ShowLabel}))

            print('##-------------------#', i)
            # here, we choose use loss value directly, which is different from other processes
            trainloss = sess.run(loss, feed_dict={x: X, y0: Y})
            testloss = sess.run(loss, feed_dict={x: testset, y0: testlabel})
            showloss = sess.run(loss, feed_dict={x: ShowSet, y0: ShowLabel})
            trloss.append(trainloss)
            teloss.append(testloss)
            shloss.append(showloss)
        ###-----------------------------------------------------------评估
            p1 = sess.run(y1, feed_dict={x: ShowSet})
            p2 = sess.run(y2, feed_dict={x: ShowSet})
            p3 = sess.run(y3, feed_dict={x: ShowSet})
            pre1.append(p1.reshape(-1))
            pre2.append(p2.reshape(-1))
            pre3.append(p3.reshape(-1))

    predict1 = np.mean(np.array(pre1), axis=0)
    predict2 = np.mean(np.array(pre2), axis=0)
    predict3 = np.mean(np.array(pre3), axis=0)
    trloss = np.mean(trloss)
    teloss = np.mean(teloss)
    shloss = np.mean(shloss)

    TT = time.time()
    print('Time', TT-T0)
    print('>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    return [predict1, predict2, predict3, trloss, teloss, shloss]

