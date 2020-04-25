# supermodel, multiprocessing
# for subset training
# logfile, './result/subsetlog.csv'
import numpy as np
import pandas as pd
from preprocess import dataprocess1, dataprocess2
from supermodel import supermodel1
from config_subset import config
import multiprocessing

# attention, the class object can't be called by function (but can inherit)
def main(config):
    import time
    T0 = time.time()

    F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,E,Kp,Dst,ap,AE = dataprocess1()
    data_in = np.hstack((F1, F2, F3, F4, F5, F6, F7, T, N, V, P, Bt, E, Kp, Dst, ap, AE))
    print('Modified data_in shape:', data_in.shape)
    data_out = dataprocess2()
    print('data_out shape:', data_out.shape)

    data_in = data_in
    Setpre = data_in
    Labelpre = data_out

    # attention! num_in is very important
    config.num_in = data_in.shape[1]

    # multiprocess training
    pool = multiprocessing.Pool(processes=7)
    ## seven years data, split into 7 parts.  2408 = 344*7
    oneyear = 344
    res = []
    res2 = []
    weightall = []
    for i in range(7):
        print('The %d year'%i)
        yrange = np.arange(oneyear*i, oneyear*i+oneyear)
        ShowSet = Setpre[yrange, :]
        ShowLabel = Labelpre[yrange, :]
        Set = np.delete(Setpre, yrange, axis=0)
        Label = np.delete(Labelpre, yrange, axis=0)
        #trloss, teloss, shloss, trainPE, testPE, showPE, trainCC, testCC, showCC, TT, \
        # weight = supermodel1(Set, Label, ShowSet, ShowLabel, config)
        res.append(pool.apply_async(supermodel1, (Set, Label, ShowSet, ShowLabel, config)))
        #res.append([trloss, teloss, shloss, trainPE, testPE, showPE, TT])
        #weightall.append(weight)
    pool.close()
    pool.join()
    print('finished !!!')
    print('res!!!', res)
    for j in res:
        x = j.get()
        res2.append(x[:-1])
        #weightall.append(x[-1])
    #--------------------------------------
    # attention! num_in is very important
    num_in = data_in.shape[1]
    variables = 'F1-F7+T+N+V+P+Bt+Ey+Kp+Dst+ap+AE'

    run_num = config.run_num
    seed = config.seed
    learnrate = config.learnrate
    epochs = config.epochs
    batchsize = config.batch_size
    reg1 = config.reg1
    reg2 = config.reg2
    node1 = config.node1
    node2 = config.node2
    f1 = config.fun1
    f2 = config.fun2

    #Ls = config.Loss
    path = './result2/'
    logfile = config.subsetlog

    res2 = pd.DataFrame(res2)
    res_mean = res2.mean(axis=0)
    res_mean = list(res_mean)
    res_mean = [np.round(x, 6) for x in res_mean]
    print('res_mean:', res_mean)

    # save data to logfile
    FFN_para = pd.read_csv(path+logfile, index_col=False, header=0, sep=',')
    config_para = [run_num, num_in, variables, seed, epochs, batchsize, learnrate, reg1, reg2,  node1, node2, f1, f2]
    finalres = config_para+res_mean
    # columns name
    col1 = ['run_num', 'num_in', 'variables', 'seed', 'epochs', 'batchsize', 'learnrate', 'reg1', 'reg2', 'node1', 'node2', 'fun1', 'fun2']
    col2 = ['trloss', 'teloss', 'shloss', 'trainPE', 'testPE', 'showPE', 'trainCC', 'testCC', 'showCC', 'TT']
    finalcol = col1+col2
    # save data
    lognum = len(FFN_para)
    FFN_para.loc[lognum] = finalres
    FFN_para.to_csv(path+logfile, index=False, header=finalcol, sep=',')

    TT = time.time()
    print('finally finished !!!', TT - T0)

if __name__=='__main__':
    config = config()
    main(config)