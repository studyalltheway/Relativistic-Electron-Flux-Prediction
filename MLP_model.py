# supermodel, multiprocessing
# logfile, './result/logfile2.csv'
# MLP model
import numpy as np
import pandas as pd
from preprocess import dataprocess1, dataprocess2
from config import config
from supermodel import supermodel2
import multiprocessing


def main(config):
    import time
    T0 = time.time()

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
    res2 = []
    for i in range(7):
        print('The %d year'%i)
        yrange = np.arange(oneyear*i, oneyear*i+oneyear)
        ShowSet = Setpre[yrange, :]
        ShowLabel = Labelpre[yrange, :]
        Set = np.delete(Setpre, yrange, axis=0)
        Label = np.delete(Labelpre, yrange, axis=0)
        #trloss, teloss, shloss, trainPE, testPE, showPE, trainCC, testCC, showCC, TT, \
        # weight = supermodel1(Set, Label, ShowSet, ShowLabel, config)
        res.append(pool.apply_async(supermodel2, (Set, Label, ShowSet, ShowLabel, config)))
        #res.append([trloss, teloss, shloss, trainPE, testPE, showPE, TT])
        #weightall.append(weight)
    pool.close()
    pool.join()
    print('finished !!!')
    print('res!!!', res)
    for j in res:
        x = j.get()
        res2.append(x)
    #--------------------------------------
    run_num = config.run_num
    assert int(run_num) >= 1000
    seed = config.seed
    epochs = config.epochs
    batchsize = config.batch_size
    learnrate = config.learnrate
    reg2 = config.reg2
    num_in = config.num_in
    node1 = config.node1
    node2 = config.node2
    f1 = config.fun1
    f2 = config.fun2
    path = './result2/'
    logfile = config.logfile2

    res2 = pd.DataFrame(res2)
    res2.to_csv('./result/MLPres'+str(run_num)+'.csv', index=False, header=None, sep=',')
    res3 = pd.read_csv('./result/MLPres'+str(run_num)+'.csv', index_col=None, header=None, skiprows=0, sep=',')
    res_mean = res3.mean(axis=0)
    res_mean = list(res_mean)
    res_mean = [np.round(x, 6) for x in res_mean]
    print('res_mean:', res_mean)

    # save data to logfile
    MLP_para = pd.read_csv(path+logfile, index_col=False, header=0, sep=',')
    config_para = [run_num, seed, epochs, batchsize, learnrate, reg2, num_in, node1, node2, f1, f2]
    finalres = config_para+res_mean
    # column name
    col1 = ['run_num', 'seed', 'epochs', 'batchsize', 'learnrate', 'reg2', 'num_in', 'node1', 'node2', 'fun1', 'fun2']
    col2 = ['trloss', 'teloss', 'shloss', 'trainPE', 'testPE', 'showPE', 'trainCC', 'testCC', 'showCC', 'TT']
    finalcol = col1+col2
    ## save data
    lognum = len(MLP_para)
    MLP_para.loc[lognum] = finalres
    MLP_para.to_csv(path+logfile, index=False, header=finalcol, sep=',')

    TT = time.time()
    print('finally finished !!!', TT - T0)
    return

if __name__=='__main__':
    config = config()
    main(config)