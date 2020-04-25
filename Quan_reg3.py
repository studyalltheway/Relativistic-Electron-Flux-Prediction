# quantile regression for predicting many quantiles at the same time
# Loss = t|f(x)-y|+(1-t)|f(x)-y|
import numpy as np
import pandas as pd
import multiprocessing
from preprocess import dataprocess1, dataprocess2
from config import config as config2
from supermodel import supermodel3_2
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


def main(config):
    F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,E,Kp,Dst,ap,AE = dataprocess1()
    data_in = np.hstack((F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,E,Kp,Dst,ap,AE))
    print('Modified data_in shape:', data_in.shape)
    data_out = dataprocess2()
    print('data_out shape:', data_out.shape)

    Setpre = data_in
    Labelpre = data_out

    # multiprocess training
    pool = multiprocessing.Pool(processes=7)
    ## seven years data, split into 7 parts.  2408 = 344*7
    oneyear = 344
    res = []
    trloss, teloss, shloss = [], [], []
    pre1, pre2, pre3, pre4, pre5, pre6, pre7, pre8, pre9 = [], [], [], [], [], [], [], [], []
    for i in range(7):
        print('The %d year'%i)
        yrange = np.arange(oneyear*i, oneyear*i+oneyear)
        ShowSet = Setpre[yrange, :]
        ShowLabel = Labelpre[yrange, :]
        Set = np.delete(Setpre, yrange, axis=0)
        Label = np.delete(Labelpre, yrange, axis=0)

        res.append(pool.apply_async(supermodel3_2, (Set, Label, ShowSet, ShowLabel, config)))

    pool.close()
    pool.join()
    print('finished !!!')
    print('res!!!', res)
    for j in res:
        x = j.get()
        pre1.append(x[0])
        pre2.append(x[1])
        pre3.append(x[2])
        pre4.append(x[3])
        pre5.append(x[4])
        pre6.append(x[5])
        pre7.append(x[6])
        pre8.append(x[7])
        pre9.append(x[8])
        trloss.append(x[9])
        teloss.append(x[10])
        shloss.append(x[11])

    ## calculate the correct rate
    rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9 = [], [], [], [], [], [], [], [], []
    for i in range(7):
        print('The %d year'%i)
        yrange = np.arange(oneyear*i, oneyear*i+oneyear)
        ShowSet = Setpre[yrange, :]
        ShowLabel = Labelpre[yrange, :]
        prediction1 = pre1[i]
        prediction2 = pre2[i]
        prediction3 = pre3[i]
        prediction4 = pre4[i]
        prediction5 = pre5[i]
        prediction6 = pre6[i]
        prediction7 = pre7[i]
        prediction8 = pre8[i]
        prediction9 = pre9[i]
        rate1.append(correct_rate(ShowLabel, prediction1))
        rate2.append(correct_rate(ShowLabel, prediction2))
        rate3.append(correct_rate(ShowLabel, prediction3))
        rate4.append(correct_rate(ShowLabel, prediction4))
        rate5.append(correct_rate(ShowLabel, prediction5))
        rate6.append(correct_rate(ShowLabel, prediction6))
        rate7.append(correct_rate(ShowLabel, prediction7))
        rate8.append(correct_rate(ShowLabel, prediction8))
        rate9.append(correct_rate(ShowLabel, prediction9))

    ###---------------
    run_num = config.run_num
    seed = config.seed
    learnrate = config.learnrate
    epochs = config.epochs
    batchsize = config.batch_size
    reg1 = config.reg1
    reg2 = config.reg2
    num_in = config.num_in
    node1 = config.node1
    node2 = config.node2
    weight_init = config.weight_init
    f1 = config.fun1
    f2 = config.fun2

    logfile = './result2/quan_log3.csv'

    para_col = ['run_num', 'seed', 'num_in', 'epochs', 'batchsize', 'learnrate', 'reg1', 'reg2', 'node1', 'node2',
            'weight_init', 'fun1', 'fun2', 'trloss', 'teloss', 'shloss']
    ratecol, ratedata = [], []
    for i in range(1, 8, 1):
        for j in range(1, 10, 1):
            ratecol.append(str(i)+'_'+str(j))
    ratedata = rate1+rate2+rate3+rate4+rate5+rate6+rate7+rate8+rate9

    finalcol = para_col+ratecol

    # save data to logfile
    Quan_para = pd.read_csv(logfile, index_col=False, header=0, sep=',')
    config_para = [run_num, seed, num_in, epochs, batchsize, learnrate, reg1, reg2, node1, node2, weight_init, f1, f2]
    finalres = config_para+[np.mean(trloss), np.mean(teloss), np.mean(shloss)]+ratedata

    lognum = len(Quan_para)

    Quan_para.loc[lognum] = finalres
    Quan_para.to_csv(logfile, index=False, header=finalcol, sep=',')

    TT = time.time()
    print('finished!!!', TT-T0)


if __name__ == "__main__":
    config = config2()
    main(config)










