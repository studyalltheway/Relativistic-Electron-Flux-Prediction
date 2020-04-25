import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class utils():
    def __int__(self):
        return

    def PE(self, y_true, y_pred):
        SV1 = np.mean(np.square(y_true - y_pred))
        SV2 = np.var(y_true)
        PE = 1 - SV1/SV2
        return PE

    def RMSE(self, y_true, y_pred):
        SV1 = np.mean(np.square(y_true - y_pred))
        RMSE = np.sqrt(SV1)
        return RMSE

    def RMAE(self, y_true, y_pred):
        SV1 = np.mean(np.abs(y_true - y_pred))
        RMAE = np.sqrt(SV1)
        return RMAE

    def MSE(self, y_true, y_pred):
        MSE = np.mean(np.square(y_true - y_pred))
        return MSE

    def MAE(self, y_true, y_pred):
        MAE = np.mean(np.abs(y_true - y_pred))
        return MAE

    def NORM(self, y_pred):
        value_mean = np.mean(y_pred)
        value_var = np.var(y_pred)
        res = (y_pred-value_mean)/value_var
        return res

class GEOplot1():
    def __init__(self):
        self.type = 1
    def Gplot1(self, y):
        fig = plt.figure()

###-------------------------------functions
def lieanercal(y_pred, label):
    """
    :param y_pred:
    :param label:
    :return: line.intercept_, line.coef_, line.correlationefficient
    """
    data1 = np.array(y_pred).reshape(-1)
    data2 = np.array(label).reshape(-1)
    linreg = LinearRegression()
    line2 = linreg.fit(data1.reshape(-1, 1), data2.reshape(-1, 1))
    Cor2 = np.corrcoef(data1, data2)[0, 1]
    print('line regression:', line2.intercept_[0], line2.coef_[0][0], Cor2)
    return line2.intercept_[0], line2.coef_[0][0], Cor2

def corrcal(y_pred, label):
    data1 = np.array(y_pred).reshape(-1)
    data2 = np.array(label).reshape(-1)
    Cor2 = np.corrcoef(data1, data2)[0, 1]
    return Cor2

def randomshuffle(Setpre, Labelpre, seed=100):
    """
    # random shuffle first, then make predictions
    :param Setpre:
    :param Labelpre:
    :param seed:
    :return: Setpre2, Labelpre2
    """
    import random
    index =np.arange(len(Setpre))
    random.seed(seed)
    random.shuffle(index)
    Setpre2 = Setpre[index]
    Labelpre2 = Labelpre[index]
    return Setpre2, Labelpre2

###---------------------metrics
def PE(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    SV1 = np.mean(np.square(y_true - y_pred))
    SV2 = np.var(y_true)
    PE = 1 - SV1/SV2
    return PE

def RMSE(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    SV1 = np.mean(np.square(y_true - y_pred))
    RMSE = np.sqrt(SV1)
    return RMSE

def RMAE(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    SV1 = np.mean(np.abs(y_true - y_pred))
    RMAE = np.sqrt(SV1)
    return RMAE

def MSE(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    MSE = np.mean(np.square(y_true - y_pred))
    return MSE

def MAE(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    MAE = np.mean(np.abs(y_true - y_pred))
    return MAE

def NORM(y_pred):
    value_mean = np.mean(y_pred)
    value_var = np.var(y_pred)
    res = (y_pred-value_mean)/np.sqrt(value_var)
    return res

def LossQ(y_true, y_pred, tau):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    error = 0
    for i in range(len(y_true)):
        if y_pred[i]-y_true[i]>0:
            error += (y_pred[i]-y_true[i])*tau
        else:
            error += (y_true[i]-y_pred[i])*(1-tau)
    return error/len(y_pred)


