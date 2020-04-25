import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='./primitive data/'
LowFlux='LowFluxHourly.csv'                              ## hourly averaged data
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
LF=pd.read_table(path+LowFlux, index_col=False, header=None, skiprows=0, sep=',')
LF=np.array(LF)
print('LowFlux:', LF.shape)

# check the information of missing data
# for j in range(5):
#     print('%d column：' % (j + 1))
#     i=0
#     sum=0
#     Flag = 0
#     while i <len(LF[:,j]):
#         if Flag==1:
#             sum=0
#         if LF[:,j][i]==0:
#             sum=sum+1
#             Flag=0
#             print(i+1,sum)
#         else:
#             Flag=1
#         i=i+1



def calzero(data):
    numzero=0
    for i in range(24):
        if data[i]==0:
            numzero+=1
    return numzero

### the first 5 columns are used to calculate sum, and the last 5 columns are used to count zeros
N=len(LF)
DF=np.zeros([int(N/24),10])
for i in range(int(N/24)):
    for j in range(5):
        DF[i,j+5]=calzero(LF[i*24:i*24+24,j])
        DF[i,j]=np.sum(LF[i*24:i*24+24,j])
print('DF---')

NDF=np.zeros([int(N/24),5])
for j in range(5):
    NDF[:,j]=DF[:,j]/(24-DF[:,5+j]+10**-8)

for i in range(int(N/24)):
    for j in range(5):
        if NDF[i,j]>=10**8:
           NDF[i,j]=0

##### check the information of missing data
# the result is there is no missing data of daily averaged data
for j in range(5):
    print('%d column：' % (j + 1))
    i=0
    sum=0
    Flag = 0
    while i <len(NDF[:,j]):
        if Flag==1:
            sum=0
        if NDF[:,j][i]==0:
            sum=sum+1
            Flag=0
            print(i+1,sum)
        else:
            Flag=1
        i=i+1

print('NDF Shape:',NDF.shape)
# insert 3 missing day data
Zero=np.zeros([1,5])
NDF=np.insert(NDF,1442,Zero,axis=0)
NDF=np.insert(NDF,1566,Zero,axis=0)
NDF=np.insert(NDF,1878,Zero,axis=0)
print('NDF Shape2:',NDF.shape)
Nh=len(NDF)


fig=plt.figure(figsize=(20, 16))
electron = ['40keV', '75keV', '150keV', '275kev', '475keV']
for j in range(5):
    plt.subplot(5,1,j+1)
    if j!=4:
        plt.xticks([])
    else:
        plt.xticks(fontsize=20)
        plt.xlabel('days', fontsize=20)
    plt.plot(np.log10(NDF[:, j]+1),'b',linewidth=2)
    plt.xlim([0, Nh])
    plt.yticks(fontsize=20)
    plt.legend([electron[j]], fontsize=20, loc='lower left')
    plt.title('total data: ' + str(Nh) + '   missing data: 3', fontsize=20)
plt.show()

# linear interpolation
for j in range(5):
    for i in range(Nh):
        if NDF[i,j]==0:
            NDF[i,j]=(NDF[i-1,j]+NDF[i+1,j])/2


savefile='LowFluxDaily.csv'
data = pd.DataFrame(NDF)
data.to_csv(path + savefile, index=False, header=None, sep=',')



