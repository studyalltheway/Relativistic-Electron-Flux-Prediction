# Figure 5 of the paper
# show one year result of the quantile regression, 2011
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from preprocess import dataprocess2
from utils import RMSE, PE, corrcal


# calculate error num
def errorcal(Label, data):
    tmp = data-Label
    tmp = [1 if x>=0 else 0 for x in tmp]
    return sum(tmp)


# calculate the error num between low and high quantiles
def errorcal2(H, L, data):
    n = len(H)
    s = 0
    for i in range(n):
        if H[i] >= data[i] and L[i] < data[i]:
            s += 1
    return s


data_out = dataprocess2()
print(data_out.shape)

Label0 = data_out.reshape(-1)
Label0 += 8



data = []
tau_range = [0.1, 0.3, 0.5, 0.7, 0.9]
for j in range(2001, 2008, 1):
    pre = []
    for x in tau_range:
        file = './result/quan_'+str(x)+'_'+str(j)+'.csv'
        predata = pd.read_csv(file, index_col=False, header=None, sep=',')
        predata = np.array(predata)[0, :]
        pre.append(predata)
    data0 = np.array(pre)+8
    data.append(data0)

data = np.mean(np.array(data), axis=0)


# determine quantiles to show
oneyear = 344
# year, 2011---0, 2012---1, ...
Y = 0
Label = Label0[oneyear*Y: oneyear*(Y+1)]


oneyear = 344
# 50-151 day
data1 = data[:, 50:151]
Label1 = Label[50:151]
# 200-300 day
data2 = data[:, 200:301]
Label2 = Label[200:301]

tmp0, tmp1, tmp2 = [], [], []
for i in range(5):
    tmp0.append(errorcal(Label[0:oneyear], data[i, 0:oneyear]))
    tmp1.append(errorcal(Label1[0:-1], data1[i, 0:-1]))
    tmp2.append(errorcal(Label2[0:-1], data2[i, 0:-1]))

dtmp1 = errorcal2(data1[0, 0:-1], data1[4, 0:-1], Label1[0:-1])
dtmp2 = errorcal2(data2[1, 0:-1], data2[3, 0:-1], Label2[0:-1])
print('error number:', tmp0, tmp1, tmp2, dtmp1, dtmp2)

# index = []
# oneyear = 344
# for i in range(7):
#     yrange = np.arange(oneyear*i, oneyear*i+oneyear)
#     predict = data[3, yrange]
#     observe = data_out[yrange]
#     index1 = RMSE(observe, predict)
#     index2 = PE(observe, predict)
#     index3 = corrcal(observe, predict)
#     index.append([index1, index2, index3])
# print('Index:', index)
# index = np.array(index)
# print('mean values:', index.mean(axis=0))

# plot the figure 5 of the paper
fsize = 20
fig = plt.figure(figsize=(20, 12))
plt.subplots_adjust(left=5, bottom=5, right=95, top=95, wspace=25, hspace=25)
fig.add_subplot(211)
plt.tick_params(labelsize=10)
plt.plot(Label, 'k-', linewidth=1.5)
plt.plot(data[0, :], 'b--', linewidth=1.2)
plt.plot(data[1, :], 'g--', linewidth=1.2)
plt.plot(data[2, :], 'r-', linewidth=1)
plt.plot(data[3, :], 'm--', linewidth=1.2)
plt.plot(data[4, :], 'c--', linewidth=1.2)
plt.legend(['observations (344)', \
            r'$\tau$=0.9, (%d, %.2f)'%(tmp0[0], tmp0[0]/344), \
            r'$\tau$=0.7, (%d, %.2f)'%(tmp0[1], tmp0[1]/344), \
            r'$\tau$=0.5, (%d, %.2f)'%(tmp0[2], tmp0[2]/344), \
            r'$\tau$=0.3, (%d, %.2f)'%(tmp0[3], tmp0[3]/344), \
            r'$\tau$=0.1, (%d, %.2f)'%(tmp0[4], tmp0[4]/344), \
            ], loc=[0.8, 0.65], fontsize=13, framealpha=0)

plt.xlim([0, 344])
plt.ylim([6, 10])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('day of 2011', fontsize=fsize)
plt.ylabel('log$_1$$_0$(flux)', fontsize=fsize)

axis1 = fig.add_subplot(223)
plt.plot(Label1, 'k-', linewidth=2)
plt.plot(data1[0], 'b--', linewidth=1.5)

plt.plot(data1[4], 'c--', linewidth=1.5)
y1, y2 = data1[0], data1[4]
plt.fill_between(np.arange(101), y1, y2, where=y1 > y2, facecolor='yellow', alpha=0.3)
plt.legend(['observations (100)', \
            r'$\tau$=0.9, (%d)'%(tmp1[0]), \
            r'$\tau$=0.1, (%d)'%(tmp1[4]), \
            r'$\Delta$$\tau$=0.8, (%d)'%(dtmp1)
            ], loc=[0.67, 0.75], fontsize=13, framealpha=0)
plt.xlim([0, 100])
plt.ylim([6, 10])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
axis1.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.xticks(np.arange(0, 110, 10), np.arange(50, 160, 10))
plt.xlabel('day of 2011', fontsize=fsize)
plt.ylabel('log$_1$$_0$(flux)', fontsize=fsize)

axis2 = fig.add_subplot(224)
plt.plot(Label2, 'k-', linewidth=2)
plt.plot(data2[1], 'g--', linewidth=1.5)

plt.plot(data2[3], 'm--', linewidth=1.5)
y1, y2 = data2[1], data2[3]
plt.fill_between(np.arange(101), y1, y2, where=y1 > y2, facecolor='yellow', alpha=0.3)
plt.legend(['observations (100)', \
            r'$\tau$=0.7, (%d)'%(tmp2[1]), \
            r'$\tau$=0.3, (%d)'%(tmp2[3]), \
            r'$\Delta$$\tau$=0.4, (%d)'%(dtmp2) \
            ], loc='upper right', fontsize=13, framealpha=0)
plt.xlim([0, 100])
plt.ylim([6, 10])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
axis2.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.xticks(np.arange(0, 110, 10), np.arange(200, 310, 10))
plt.xlabel('day of 2011', fontsize=fsize)
plt.ylabel('log$_1$$_0$(flux)', fontsize=fsize)

plt.show()


