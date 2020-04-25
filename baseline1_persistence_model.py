# baseline, persistence model
import numpy as np
from utils import RMSE, PE, corrcal
from preprocess import dataprocess2

label = dataprocess2()
# true label
label += 8
print('label shape:', label.shape)

oneyear = 344
srange = np.arange(344)
rmse, pe, lc = [], [], []
for i in range(7):
    label2 = label[oneyear*i+srange]
    trueflux = label2[1:]
    persistence = label2[0:-1]
    rmse.append(RMSE(trueflux, persistence))
    pe.append(PE(trueflux, persistence))
    lc.append(corrcal(trueflux, persistence))
rmse = np.mean(rmse)
print(pe)
pe = np.mean(pe)
lc = np.mean(lc)
print('Result: RMSE %f, PE %f, LC %f'%(rmse, pe, lc))