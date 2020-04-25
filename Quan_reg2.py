# quantile regression for predicting three quantiles at the same time
# Loss = t|f(x)-y|+(1-t)|f(x)-y|
import numpy as np
import pandas as pd
import multiprocessing
from preprocess import dataprocess1, dataprocess2
from config import config as config2
from supermodel import supermodel3_3
import time
T0 = time.time()


def correct_rate(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    right_num = 0
    for i in range(len(y_true)):
        if y_true[i] <= y_pred[i]:
            right_num += 1
    return float(right_num)/len(y_true)


def alpha2cal(Setpre, Labelpre, config):
    T0 = time.time()
    pool = multiprocessing.Pool(processes=7)

    oneyear = 344
    res, pre1, pre2, pre3 = [], [], [], []
    trainloss, testloss, showloss = [], [], []
    for i in range(7):
        print('The %d year' % i)
        yrange = np.arange(oneyear * i, oneyear * i + oneyear)
        ShowSet = Setpre[yrange, :]
        ShowLabel = Labelpre[yrange, :]
        Set = np.delete(Setpre, yrange, axis=0)
        Label = np.delete(Labelpre, yrange, axis=0)
        res.append(pool.apply_async(supermodel3_3, (Set, Label, ShowSet, ShowLabel, config)))

    pool.close()
    pool.join()

    for j in res:
        x = j.get()
        pre1.append(x[0])
        pre2.append(x[1])
        pre3.append(x[2])
        trainloss.append(x[3])
        testloss.append(x[4])
        showloss.append(x[5])

    pre1 = np.array(pre1).reshape(-1)
    pre2 = np.array(pre2).reshape(-1)
    pre3 = np.array(pre3).reshape(-1)
    prediction = np.vstack((pre1, pre2, pre3))

    trainloss = np.mean(trainloss)
    testloss = np.mean(testloss)
    showloss = np.mean(showloss)

    TT = time.time()
    print('finished!', TT - T0)
    return prediction, trainloss, testloss, showloss

if __name__=='__main__':
    F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,E,Kp,Dst,ap,AE = dataprocess1()
    data_in = np.hstack((F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,E,Kp,Dst,ap,AE))
    print('Modified data_in shape:', data_in.shape)
    data_out = dataprocess2()
    print('data_out shape:', data_out.shape)

    Setpre = data_in
    Labelpre = data_out

    # model prediction
    config = config2()
    # if hasattr(config, 'taurange'):
    #     config.taurange = [0.3, 0.5, 0.8]
    prediction, trainloss, testloss, showloss = alpha2cal(Setpre, Labelpre, config)


    # save result to log file
    logfile = './result2/quan_reg2.csv'

    para_col = ['run_num', 'tau_range', 'seed', 'pterm', 'num_in', 'epochs', 'batchsize', 'learnrate', 'reg1', 'reg2', 'node1', 'node2',
            'trloss', 'teloss', 'shloss', 'showrate1', 'showrate2', 'showrate3']

    # parameters
    run_num = config.run_num
    tau_range = config.taurange
    tau_range = str(tau_range[0])+'_'+str(tau_range[1])+'_'+str(tau_range[2])
    seed = config.seed
    pterm = config.pterm
    num_in = config.num_in
    epochs = config.epochs
    batchsize = config.batch_size
    learnrate = config.learnrate
    reg1 = config.reg1
    reg2 = config.reg2
    node1 = config.node1
    node2 = config.node2

    oneyear = 344
    showrate1, showrate2, showrate3 = [], [], []
    for i in range(7):
        print('The %d year'%i)
        yrange = np.arange(oneyear*i, oneyear*i+oneyear)
        ShowLabel = Labelpre[yrange, :]
        showrate1.append(correct_rate(ShowLabel, prediction[0, yrange]))
        showrate2.append(correct_rate(ShowLabel, prediction[1, yrange]))
        showrate3.append(correct_rate(ShowLabel, prediction[2, yrange]))

    showrate1 = np.mean(showrate1)
    showrate2 = np.mean(showrate2)
    showrate3 = np.mean(showrate3)

    trainloss = np.round(trainloss, 6)
    testloss = np.round(testloss, 6)
    showloss = np.round(showloss, 6)
    showrate1 = np.round(showrate1, 6)
    showrate2 = np.round(showrate2, 6)
    showrate3 = np.round(showrate3, 6)

    # save data to logfile
    Quan_para = pd.read_csv(logfile, index_col=False, header=0, sep=',')
    config_para = [run_num, tau_range, seed, pterm, num_in, epochs, batchsize, learnrate, reg1, reg2, node1, node2, trainloss, testloss, showloss, showrate1, showrate2, showrate3]
    finalres = config_para
    print(config_para)

    finalcol = para_col

    lognum = len(Quan_para)

    Quan_para.loc[lognum] = finalres
    Quan_para.to_csv(logfile, index=False, header=finalcol, sep=',')

    TT = time.time()
    print('finished!!!', TT-T0)










