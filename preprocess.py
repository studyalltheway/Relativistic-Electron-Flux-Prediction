import numpy as np
import pandas as pd
from utils import RMSE, PE, NORM
from config import config


def dataprocess1():
    con = config()
    file3 = con.datapath+con.set3
    file4 = con.datapath+con.set4
    file5 = con.datapath+con.set5
    print('file:', file3, '\n', file4, '\n', file5)
    Set3 = pd.read_csv(file3, index_col=False, header=None, skiprows=0, sep=',')
    Set3 = np.array(Set3)
    print('Set3 shape:', Set3.shape)
    Set4 = pd.read_csv(file4, index_col=False, header=None, skiprows=0, sep=',')
    Set4 = np.array(Set4)
    print('Set4 shape:', Set4.shape)
    Set5 = pd.read_csv(file5, index_col=False, header=None, skiprows=0, sep=',')
    Set5 = np.array(Set5)
    print('Set5 shape:', Set5.shape)
    # S2set3=[]   # high and medium energy electron flux
    # S2set4=[]   ####Kp Dst ap AE
    # S2set5=[]   ####Bt Bx By Bz T N V P E
    ####--------------------------------------make datasets
    Set4 = Set4.reshape(-1, 5, 4)
    Kp = Set4[:, :, 0]
    Dst = Set4[:, :, 1]
    ap = Set4[:, :, 2]
    AE = Set4[:, :, 3]
    # S2set5=[]   #######Bt Bx By Bz T N V P E
    Set5 = Set5.reshape(-1, 5, 9)
    Bt = Set5[:, :, 0]
    # Bx = Set5[:, :, 1]
    # By = Set5[:, :, 2]
    # Bz = Set5[:, :, 3]
    T = Set5[:, :, 4]
    N = Set5[:, :, 5]
    V = Set5[:, :, 6]
    P = Set5[:, :, 7]
    Ey = Set5[:, :, 8]

    Set3=Set3.reshape(-1, 5, 7)
    F1 = Set3[:, :, 1]
    F2 = Set3[:, :, 0]
    F3 = Set3[:, :, 6]
    F4 = Set3[:, :, 5]
    F5 = Set3[:, :, 4]
    F6 = Set3[:, :, 3]
    F7 = Set3[:, :, 2]

    for i in range(5):
        F1[:, i] = NORM(F1[:, i])
        F2[:, i] = NORM(F2[:, i])
        F3[:, i] = NORM(F3[:, i])
        F4[:, i] = NORM(F4[:, i])
        F5[:, i] = NORM(F5[:, i])
        F6[:, i] = NORM(F6[:, i])
        F7[:, i] = NORM(F7[:, i])
        V[:, i] = NORM(V[:, i])
        Ey[:, i] = NORM(Ey[:, i])
        T[:, i] = NORM(T[:, i])
        Bt[:, i] = NORM(Bt[:, i])
        N[:, i] = NORM(N[:, i])
        # By[:, i] = NORM(By[:, i])
        # Bz[:, i] = NORM(Bz[:, i])
        # Bx[:, i] = NORM(Bx[:, i])
        P[:, i] = NORM(P[:, i])
        Kp[:, i] = NORM(Kp[:, i])
        AE[:, i] = NORM(AE[:, i])
        Dst[:, i] = NORM(Dst[:, i])
        ap[:, i] = NORM(ap[:, i])
    return F1, F2, F3, F4, F5, F6, F7,  T, N, V, P, Bt, Ey,  Kp, Dst, ap, AE


def dataprocess2():
    con = config()
    label1 = con.datapath+con.label1
    print('label file:', label1)
    Label1 = pd.read_csv(label1, index_col=False, header=None, skiprows=0, sep=',')
    Label1 = np.array(Label1)
    ## Attention ！！！
    ## because the unit is log10 cm-2d-1sr-1, the label need to be adjusted
    ## the primitive unit is log10 cm-2s-1sr-1, one of 60*60*24  of the formal unit,
    ## so the values of label should be added by log10(60*60*24) = 4.936514
    ## the F1, F2, F3, F4, F5, F6, F7 don't need such an adjustment, because they are normalized to mean value of zero.
    # true value
    Label1 += 4.936514
    ## Attention 2 ！！！
    ## Becanse the neural network are easier to predict a value near zero, the predict value is ture value - 8
    ## 8 is near the mean of the Label1
    Label1 -= 8
    print('Label1 shape:', Label1.shape)
    return Label1