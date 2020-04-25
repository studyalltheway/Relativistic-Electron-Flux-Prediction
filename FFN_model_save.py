# supermodel1_2, multiprocessing, save the prediction data into result2 file
import numpy as np
import pandas as pd
from preprocess import dataprocess1, dataprocess2
from supermodel import supermodel1_2
from config import config
import multiprocessing

# attention, the class object can't be called by function (but can inherit)
def main(config):
    import time
    T0 = time.time()

    F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,Bx,By,Bz,E,Kp,Dst,ap,AE = dataprocess1()
    data_in = np.hstack((F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,Bx,By,Bz,E,Kp,Dst,ap,AE))
    print('Modified data_in shape:', data_in.shape)
    data_out = dataprocess2()
    print('data_out shape:', data_out.shape)

    data_in = data_in
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
        res.append(pool.apply_async(supermodel1_2, (Set, Label, ShowSet, ShowLabel, config)))

    pool.close()
    pool.join()
    print('finished !!!')
    print('res!!!', res)
    for j in res:
        x = j.get()
        res2.append(x)
    #--------------------------------------
    seed = config.seed
    res2 = pd.DataFrame(res2)
    print('res2 shape:', res2)
    path = './result2/'
    res2.to_csv(path+'FFN_prediction'+str(seed)+'.csv', index=False, header=None, sep=',')

    TT = time.time()
    print('finally finished !!!', TT - T0)

if __name__=='__main__':
    config = config()
    main(config)