# figure 4 in subsection 4-1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import dataprocess2
from sklearn.linear_model import LinearRegression
from utils import PE, MSE, RMSE, corrcal

Label = dataprocess2()
Label = np.array(Label)
Label += 8
print('Label shape:', Label.shape)

# read the prediction data
# file name is like 'FFN_prediction2.csv', 2 is the seed
path = './result2/'
pre = []
for i in range(10):
    data = pd.read_csv(path+'FFN_prediction'+str(i+1)+'.csv', index_col=False, header=None, sep=',')
    data = np.array(data)
    pre.append(data)
pre = np.mean(np.array(pre), axis=0)
print('prediction shape:', pre.shape)
# true value
pre += 8

# plot the results
fig = plt.figure(figsize=(20, 6.8))
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
#fig.subplots_adjust(left=0, right=1, wspace=0, hspace=0)
plt.subplots_adjust(wspace=0, hspace=0.4)
res = []
oneyear = 344
for i in range(7):
    fluxtrue = Label[i*oneyear: (i+1)*oneyear].reshape(-1, 1)
    prediction = pre[i, :].reshape(-1, 1)
    PE1=PE(fluxtrue, prediction)
    RMSE1=RMSE(fluxtrue, prediction)
    linreg = LinearRegression()
    line1=linreg.fit(fluxtrue, prediction)
    Cor1=np.corrcoef(fluxtrue.reshape(-1), prediction.reshape(-1))[0, 1]
    res.append([RMSE1, PE1, Cor1])
    #### figure plot
    fsize = 15
    xx = np.arange(5.5, 10, 0.1)

    # plot the flux
    fig.add_subplot(2, 8, i+1)
    plt.plot(fluxtrue, 'b', linewidth=0.8)
    #plt.plot(prediction, 'r', linewidth=0.8)
    plt.xlim([0, 344])
    plt.ylim([6, 10.5])
    plt.xticks([])
    plt.xlabel('\n 201'+str(i+1), fontsize=fsize)
    plt.title('Mean:' + str(round(np.mean(fluxtrue), 3)) + '  Var:' + str(round(np.var(fluxtrue), 3)), fontsize=11)
    if i == 0:
        plt.ylabel('log${_1}{_0}$(flux)', fontsize=fsize)
        plt.legend(['observations', 'predictions'], fontsize=13, loc='upper right', framealpha=0)
    else:
        plt.yticks([])

    # plot the fitting result
    fig.add_subplot(2, 8, 9+i)
    if i != 0:
        plt.yticks([])
    plt.xlim([5, 10.5])
    plt.ylim([5, 10.5])
    if i == 0:
        plt.xlabel('observation / $log{_1}{_0}$(flux)', fontsize=12)
        plt.ylabel('prediction / $log{_1}{_0}$(flux)', fontsize=12)
    else:
        pass
       #plt.xlabel('201'+str(i+1), fontsize=15)
    plt.plot(xx, xx * line1.coef_[0] + line1.intercept_, 'k--')
    plt.plot(fluxtrue, prediction, 'ro', markersize=1)

    plt.title('RMSE:' + str(round(RMSE1, 4)) + '  PE:' + str(round(PE1, 4)), fontsize=11)

    ax = np.round(line1.coef_[0], 3)
    bx = np.round(line1.intercept_, 3)
    c=np.round(Cor1, 4)
    #plt.legend(['y='+str(ax[0])+'x + '+str(bx[0])+'\n'+'LC='+str(c)],loc='upper left',fontsize=9, framealpha=0)
    plt.legend(['LC='+str(c)], loc='upper left', fontsize=11, framealpha=0)

res = np.mean(np.array(res), axis=0)
print('The average result:', res)

fig.add_subplot(2, 8, 8)
plt.hist(Label, bins=20, density=True, color='b')
ax = plt.gca()
ax.yaxis.set_ticks_position('right')
plt.xlabel('$log{_1}{_0}$(flux)\n 2011-2017', fontsize=13)
#plt.ylabel('Probability')
plt.ylim([0, 0.6])
plt.text(11, 0.2, 'Probability', rotation=90, fontsize=12)
plt.title('Mean:' + str(round(np.mean(Label), 3)) + '  Var:' + str(round(np.var(Label), 3)), fontsize=11)


fig.add_subplot(2, 8, 16)
plt.text(0.2, 0.4, 'RMSE: %.4f\n PE:  %.4f\n LC: %.4f'%(res[0], res[1], res[2]), fontsize=15)
plt.xticks([])
plt.yticks([])
plt.xlabel('\n 2011-2017  Average', fontsize=13)

#plt.tight_layout()  ### Attention!
plt.show()


