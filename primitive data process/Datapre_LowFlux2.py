import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# calculate the mean value from 9 telescopes data
path='./primitive data/'
LowFlux='LowFlux.csv'                              ####hourly averaged data
#####====================
## data infomation
#2011 no missing data
#2012 no missing data
#2013 miss 18 day data (from May 23th to June 09th)
#2014 miss 1 day data (Dec. 31th)
#2015 miss 1 day data, May 4th
#2016 miss 1 day data，March 11th
#2017 miss 17 day data, from Dec. 15th to Dec. 31th
#####====================
LF=pd.read_table(path+LowFlux, header=None, skiprows=0,sep=',')
LF=np.array(LF)
print('LowFlux:',LF.shape)

fig=plt.figure()
plt.subplot(5,1,1)
plt.plot(np.log10(LF[:,2]+1),'b',linewidth=0.5)
plt.subplot(5,1,2)
plt.plot(np.log10(LF[:,11]+1),'b',linewidth=0.5)
plt.subplot(5,1,3)
plt.plot(np.log10(LF[:,20]+1),'b',linewidth=0.5)
plt.subplot(5,1,4)
plt.plot(np.log10(LF[:,29]+1),'b',linewidth=0.5)
plt.subplot(5,1,5)
plt.plot(np.log10(LF[:,38]+1),'b',linewidth=0.5)
plt.show()

# calculate the hourly averaged data
N=len(LF)
NLF=np.zeros([N,45])
for j in range(45):
    NLF[:,j]=LF[:,j]/(12-LF[:,j+45]+10**-8)

for i in range(N):
    for j in range(45):
        if NLF[i,j]>=10**8:
            NLF[i,j]=0


for j in range(45):
    sum=0
    for i in range(N):
        if NLF[i,j]==0:
            sum=sum+1
    print('%d column:'%(j+1),sum)                 #####117

NN=np.zeros([N,5])
for i in range(N):
    for j in range(5):
        NN[i,j]=np.sum(NLF[i,j*9:j*9+9])
for j in range(5):                             #####chech if there is an error
    sum = 0
    for i in range(N):
        if NLF[i, j] == 0:
            sum = sum + 1
    print('%d column:' % (j + 1), sum)            #####all 117，no error

savefile='LowFluxHourly.csv'
data = pd.DataFrame(NN)
data.to_csv(path + savefile, index=False, header=None, sep=',')
print('finished!')



