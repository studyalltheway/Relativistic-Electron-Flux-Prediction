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


def alpha2cal(Setpre, Labelpre, config):
    T0 = time.time()
    pool = multiprocessing.Pool(processes=7)

    oneyear = 344
    res, pre1, pre2, pre3 = [], [], [], []
    loss, totalloss = [], []
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

    pre1 = np.array(pre1).reshape(-1)
    pre2 = np.array(pre2).reshape(-1)
    pre3 = np.array(pre3).reshape(-1)
    prediction = np.vstack((pre1, pre2, pre3))

    TT = time.time()
    print('finished!', TT - T0)
    return prediction

if __name__=='__main__':
    F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,E,Kp,Dst,ap,AE = dataprocess1()
    data_in = np.hstack((F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,E,Kp,Dst,ap,AE))
    print('Modified data_in shape:', data_in.shape)
    data_out = dataprocess2()
    print('data_out shape:', data_out.shape)

    Setpre = data_in
    Labelpre = data_out

    for j in range(10):
        pre = []
        config = config2()
        config.seed += 1
        for i in range(1, 10, 1):
            if hasattr(config, 'taurange'):
                config.taurange = [0.5-0.05*i, 0.5, 0.5+0.05*i]
            prediction = alpha2cal(Setpre, Labelpre, config)
            pre.append(prediction)

        seed = config.seed
        pre = np.array(pre).reshape(27, -1)
        prediction = pd.DataFrame(pre)
        prediction.to_csv('./result2/Quan_'+str(seed)+'.csv', index=False, header=None, sep=',')

    TT = time.time()
    print('finished!', TT - T0)










