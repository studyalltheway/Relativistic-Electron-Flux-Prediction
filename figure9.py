# combination test7 and test8, plot 3 figures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import dataprocess2

Label = dataprocess2()
oneyear = 344
data2011true = Label[0:344] + 8
data2014true = Label[oneyear*3:oneyear*4] + 8
data2017true = Label[oneyear*6:oneyear*7] + 8
print(data2011true.shape)

file = './result2/quan_log3.csv'
data = pd.read_csv(file, index_col=False, header=0, sep=',')
print(data.shape)
data = data.iloc[:, 16:]
print(data.shape)
data = np.array(data)

data = data[8, :]
data = data.reshape(9, 7)

# plot the figure
fig = plt.figure(figsize=(20, 20))
#fig.subplots_adjust(left=2, right=20, wspace=7, hspace=0)
fsize = 20
xx = np.arange(0.1, 1, 0.1)
# plot the reliability diagram
plt.subplot(2, 2, 1)
error = []
color = ['r', 'b', 'g', 'y', 'c', 'coral', 'm']
for i in range(7):
    plt.plot(xx, data[:, i][::-1], 'o-', color=color[i], markersize=13)
    error.append(np.mean(np.abs(xx[::-1] - data[:, i])))
print('error:', np.mean(error))
xx = np.arange(0, 1, 0.01)
plt.plot(xx, xx, 'k-')
plt.xlim([0, 1])
plt.ylim([0, 1])

plt.xlabel('predicted frequency '+r'($\tau$)', fontsize=fsize)
plt.ylabel('observed frequency', fontsize=fsize)
plt.title('Reliability diagram', fontsize=fsize)
leg = ['201'+str(i)+'  ('+str(round(error[i-1], 3))+')' for i in range(1, 8)]
leg += ['2011 ('+str(np.round(np.mean(error), 3))+')']
plt.legend(leg, fontsize=fsize, framealpha=1)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=fsize)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=fsize)
plt.grid()


file = './result2/quan9.csv'
data = pd.read_csv(file, index_col=False, header=None, sep=',')
print(data.shape)
data = np.array(data)

def prob(data):
    pro = []
    for i in range(8):
        height = 0.1/(data[i+1]-data[i])
        pro.append(height)
    return pro


data2011 = []
data2014 = []
data2017 = []
for i in range(9):
    data2011.append(data[i*7])
    data2014.append(data[i*7+3])
    data2017.append(data[i*7+6])
data2011 = np.array(data2011) + 8
data2014 = np.array(data2014) + 8
data2017 = np.array(data2017) + 8

# plot the flux distribution of a single day
plt.subplot(2, 2, 2)
for i in range(0, 5):
    daydata = data2011[:, i][::-1]
    pro = prob(daydata)
    print('pro:', pro)
    width = daydata[1:]-daydata[0:-1]
    color = ['r', 'b', 'g', 'y', 'm']
    for j in range(8):
        plt.bar((daydata[j+1]+daydata[j])/2, height=pro[j], width=daydata[j+1]-daydata[j], color=color[i%5], align='center', alpha=0.5)
plt.xlabel('log${_1}{_0}$(flux)', fontsize=fsize)
plt.ylabel('Probability', fontsize=fsize)
plt.title('flux distribution prediction of a single day(2011, 1-5th day)', fontsize=fsize)
plt.xlim([7.5, 9])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)


# plot the cumulative flux distribution of 2011
plt.subplot(2, 2, 3)
data2011 = data2011.reshape(-1)
plt.hist(data2011, bins=40, density=True, color='r', alpha=0.5)
plt.hist(data2011true, bins=40, density=True, color='b', alpha=0.5)
plt.legend(['quantile regression', 'observations'], fontsize=fsize, framealpha=1)
plt.ylim([0, 1.2])
plt.xlim([6, 10])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel('log${_1}{_0}$(flux)', fontsize=fsize)
plt.ylabel('Probability', fontsize=fsize)
plt.title('2011', fontsize=fsize)

# plot the cumulative flux distribution of 2014
plt.subplot(2, 2, 4)
data2014 = data2014.reshape(-1)
plt.hist(data2014, bins=40, density=True, color='r', alpha=0.5)
plt.hist(data2014true, bins=40, density=True, color='b', alpha=0.5)
plt.legend(['quantile regression', 'observations'], fontsize=fsize, framealpha=1)
plt.ylim([0, 1.2])
plt.xlim([6, 10])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel('log${_1}{_0}$(flux)', fontsize=fsize)
plt.ylabel('Probability', fontsize=fsize)
plt.title('2014', fontsize=fsize)


plt.show()







