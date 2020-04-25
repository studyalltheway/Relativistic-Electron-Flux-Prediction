# Figure 7 of the paper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import dataprocess2


def correct_rate(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    right_num = 0
    for i in range(len(y_true)):
        if y_true[i] <= y_pred[i]:
            right_num += 1
    return float(right_num)/len(y_true)


Label = dataprocess2()
Label += 8
print(Label.shape)

##  single value prediction of quantile
RIdata = pd.read_csv('./result2/quan_log.csv', index_col=None, header=0, skiprows=268, nrows=133, sep=',')
RIdata = np.array(RIdata)[:, -16:].reshape(-1, 16*19)
print(RIdata.shape)
RIdata = np.mean(RIdata, axis=0).reshape(-1, 16)


pre = []
for i in range(11, 20):
    file = 'Quan_'+str(i)+'.csv'
    # 9*3*2408
    predata = pd.read_csv('./result2/'+file, index_col=False, header=None, sep=',')
    predata = np.array(predata)
    predata += 8
    pre.append(predata)

print(pre[0].shape)
pre = np.mean(np.array(pre), axis=0)
print('pre shape:', pre.shape)

oneyear = 344
datayear = []
rateyear = []
errorrate = []
for i in range(7):
       print('The %d year'%i)
       yrange = np.arange(oneyear*i, oneyear*i+oneyear)
       datayear = pre[:, yrange]
       for i in range(datayear.shape[0]):
           rate = correct_rate(Label[yrange], datayear[i, :])
           rateyear.append(rate)
       for i in range(len(datayear)//3):
            errorrate.append(correct_rate(datayear[i, :], datayear[i+1, :]))
            errorrate.append(correct_rate(datayear[i+1, :], datayear[i+2, :]))

print('errorate:', len(errorrate), errorrate)

data2011 = rateyear[0:27]
data2012 = datayear[27:54]
data2013 = datayear[54:81]
data2014 = datayear[81:108]
data2015 = datayear[108:135]
data2016 = datayear[135:162]
data2017 = datayear[162:189]

print(rateyear)

xx = np.arange(0.05, 1, 0.05)

fig = plt.figure(figsize=(21, 7))
plt.subplots_adjust(wspace=0, hspace=0.4)
plt.tick_params(labelsize=12)

plt.subplot(1, 3, 1)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(np.arange(0, 1, 0.1), fontsize=20)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=20)
plt.xlabel('predicted frequency', fontsize=20)
plt.ylabel('observed frequency', fontsize=20)


xx = np.arange(0, 1.05, 0.05)
plt.plot(xx, xx, 'k-', linewidth=2)
xx2 = np.arange(0, 0.5, 0.01)
plt.plot(xx2, 0.5*np.ones(len(xx2)), 'k--', linewidth=2)
plt.plot(0.5*np.ones(len(xx2)), xx2, 'k--', linewidth=2)

#color = ['coral', 'b', 'g', 'y', 'r', 'c', 'm', 'pink', 'Lime']
color = ['b', 'g', 'y', 'tan', 'coral', 'c', 'pink', 'Lime', 'm']
for i in range(9, 0, -1):
    tau_range = [0.5+0.05*i, 0.5, 0.5-0.05*i]
    plt.plot(tau_range, rateyear[3*i-3: 3*i], color=color[i-1], marker='o', linewidth=1, markersize=10)
    yy1 = [0, rateyear[3*i-3: 3*i][0], 0.0001]
    yy2 = [0, rateyear[3*i-3: 3*i][2], 0.0001]
    plt.plot(np.ones([len(yy1)])*tau_range[0], yy1, color=color[i-1], linewidth=1)
    plt.plot(np.ones([len(yy2)])*tau_range[2], yy2, color=color[i-1], linewidth=1)

plt.title('2011', fontsize=20)
xx0 = np.arange(0.05, 1, 0.05)
# RIdata[:, 3] 2011; RIdata[:, 6+3] 2014; RIdata[:, 12+3] 2017;
plt.plot(xx0[::-1], RIdata[:, 3], 'rv--', markersize=6, linewidth=0.5)
plt.grid()


plt.subplot(1, 3, 2)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(np.arange(0, 1, 0.1), fontsize=20)
plt.yticks([])
plt.xlabel('predicted frequency', fontsize=20)

xx = np.arange(0, 1.05, 0.05)
plt.plot(xx, xx, 'k-', linewidth=2)
xx2 = np.arange(0, 0.5, 0.01)
plt.plot(xx2, 0.5*np.ones(len(xx2)), 'k--', linewidth=2)
plt.plot(0.5*np.ones(len(xx2)), xx2, 'k--', linewidth=2)

color = ['b', 'g', 'y', 'tan', 'coral', 'c', 'pink', 'Lime', 'm']
for i in range(9, 0, -1):
    tau_range = [0.5+0.05*i, 0.5, 0.5-0.05*i]
    plt.plot(tau_range, rateyear[81+3*i-3: 81+3*i], color=color[i-1], marker='o', linewidth=1, markersize=10)
    yy1 = [0, rateyear[81+3*i-3: 81+3*i][0], 0.0001]
    yy2 = [0, rateyear[81+3*i-3: 81+3*i][2], 0.0001]
    plt.plot(np.ones([len(yy1)])*tau_range[0], yy1, color=color[i-1], linewidth=1)
    plt.plot(np.ones([len(yy2)])*tau_range[2], yy2, color=color[i-1], linewidth=1)

plt.title('2014', fontsize=20)
plt.plot(xx0[::-1], RIdata[:, 6+3], 'rv--', markersize=6, linewidth=0.5)
plt.grid()
#  plot the grid line
xg = np.arange(0, 1.01, 0.01)
yg = np.ones([len(xg)])
for i in range(9):
    plt.plot(xg, yg*0.1*(i+1), 'grey', linewidth=0.5)

plt.subplot(1, 3, 3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=20)
plt.yticks([])
plt.xlabel('predicted frequency', fontsize=20)


xx = np.arange(0, 1.05, 0.05)
plt.plot(xx, xx, 'k-', linewidth=2)
xx2 = np.arange(0, 0.5, 0.01)
plt.plot(xx2, 0.5*np.ones(len(xx2)), 'k--', linewidth=2)
plt.plot(0.5*np.ones(len(xx2)), xx2, 'k--', linewidth=2)

#color = ['coral', 'b', 'g', 'y', 'r', 'c', 'm', 'pink', 'Lime']
color = ['b', 'g', 'y', 'tan', 'coral', 'c', 'pink', 'Lime', 'm']
for i in range(9, 0, -1):
    tau_range = [0.5+0.05*i, 0.5, 0.5-0.05*i]
    plt.plot(tau_range, rateyear[162+3*i-3: 162+3*i], color=color[i-1], marker='o', linewidth=1, markersize=10)
    yy1 = [0, rateyear[162+3*i-3: 162+3*i][0], 0.0001]
    yy2 = [0, rateyear[162+3*i-3: 162+3*i][2], 0.0001]
    plt.plot(np.ones([len(yy1)])*tau_range[0], yy1, color=color[i-1], linewidth=1)
    plt.plot(np.ones([len(yy2)])*tau_range[2], yy2, color=color[i-1], linewidth=1)

plt.title('2017', fontsize=20)
plt.plot(xx0[::-1], RIdata[:, 12+3], 'rv--', markersize=6, linewidth=0.5)
plt.grid()
#  plot the grid line
xg = np.arange(0, 1.01, 0.01)
yg = np.ones([len(xg)])
for i in range(9):
    plt.plot(xg, yg*0.1*(i+1), 'grey', linewidth=0.5)

plt.show()