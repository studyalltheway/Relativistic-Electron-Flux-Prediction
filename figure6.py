# Figure 7 of the paper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RIdata = pd.read_csv('./result2/quan_log.csv', index_col=None, header=0, skiprows=268, nrows=133, sep=',')
RIdata = np.array(RIdata)[:, -16:].reshape(-1, 16*19)
print(RIdata.shape)
RIdata = np.mean(RIdata, axis=0).reshape(-1, 16)

xx = np.arange(0.05, 1, 0.05)

fig = plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=12)

#plt.errorbar(xx, mres, yerr=dres, color='red')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('predicted frequency '+r'($\tau$)', fontsize=20)
plt.ylabel('observed frequency', fontsize=20)

#color = ['coral', 'b', 'g', 'Lime', 'y', 'c', 'm', 'pink']
color = ['r', 'b', 'g', 'y', 'c', 'coral', 'm']
error = []
for i in range(7):
    plt.plot(xx[::-1], RIdata[:, i*2+3], 'o--', color=color[i], markersize=8)
    error.append(np.mean(np.abs(xx[::-1]-RIdata[:, i*2+3])))

# plot the diagonal
xx = np.arange(0, 1.05, 0.05)
plt.plot(xx, xx, 'k-', linewidth=2)

leg = ['201'+str(i)+'  ('+str(round(error[i-1], 3))+')' for i in range(1, 8)]
leg += ['2011 (0.050)']
plt.legend(leg, loc='upper left', fontsize=20, framealpha=1)

plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid()

# average error of quantile regression
print('error:', np.mean(error))

plt.show()