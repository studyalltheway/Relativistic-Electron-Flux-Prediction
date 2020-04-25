# baseline, multiple regression model
import numpy as np
from utils import RMSE, PE, corrcal
from preprocess import dataprocess1, dataprocess2


# multiple linear regression model
def superlinear(Set, Label, ShowSet, ShowLabel):
    from sklearn.linear_model import LinearRegression

    pre = []
    ## model training
    splitnum = len(Set)//10
    for i in range(10):
        # make data sets
        srange=np.arange(splitnum*i, splitnum*i+splitnum)
        #testset=Set[srange, :]
        #testlabel=Label[srange, :]
        trainset=np.delete(Set, srange, axis=0)
        trainlabel=np.delete(Label, srange, axis=0)

        ## build the model
        lmodel = LinearRegression()
        lmodel.fit(trainset, trainlabel)
        pre.append(lmodel.predict(ShowSet))

    pre = np.mean(np.array(pre), axis=0)
    rmse = RMSE(ShowLabel, pre)
    pe = PE(ShowLabel, pre)
    lc = corrcal(ShowLabel, pre)
    return rmse, pe, lc


F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,E,Kp,Dst,ap,AE = dataprocess1()
data_in = np.hstack((F1,F2,F3,F4,F5,F6,F7,T,N,V,P,Bt,E,Kp,Dst,ap,AE))
print('Modified data_in shape:', data_in.shape)
data_out = dataprocess2()
# true value
data_out += 8
print('data_out shape:', data_out.shape)
Setpre = data_in
Labelpre = data_out

oneyear = 344
srange = np.arange(344)
rmse, pe, lc = [], [], []
for i in range(7):
    print('The %d year'%i)
    yrange = np.arange(oneyear*i, oneyear*i+oneyear)
    ShowSet = Setpre[yrange, :]
    ShowLabel = Labelpre[yrange, :]
    Set = np.delete(Setpre, yrange, axis=0)
    Label = np.delete(Labelpre, yrange, axis=0)
    a, b, c = superlinear(Set, Label, ShowSet, ShowLabel)
    rmse.append(a)
    pe.append(b)
    lc.append(c)

rmse = np.mean(rmse)
pe = np.mean(pe)
lc = np.mean(lc)
print('Result: RMSE %f, PE %f, LC %f'%(rmse, pe, lc))