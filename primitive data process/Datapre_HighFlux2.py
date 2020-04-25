import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# calculate the daily averaged data
path = './primitive data/'
HighFF = 'HighFF.csv'
#####====================
## data infomation
#2011 no missing data
#2012 no missing data
#2013 miss 31 data (May)
#2014 miss 1 day data (Dec. 31th)
#2015 miss 1 day data, May 4th
#2016 miss 2 day data，March 11th and August 31th
#2017 miss 17 day data, from Dec. 15th to Dec. 31th
#####====================
FF=pd.read_table(path+HighFF, index_col=False, header=None, skiprows=0, sep=',')
FF=np.array(FF)
print('FF:', FF.shape)

def calzero(data):
    numzero = 0
    for i in range(24):
        if data[i] == 0:
            numzero += 1
    return numzero

### the first 2 columns are used to calculate sum, and the last 2 columns are used to count zeros
N=len(FF)
DF=np.zeros([int(N/24), 4])
for i in range(int(N/24)):
    for j in range(2):
        DF[i,j+2]=calzero(FF[i*24:i*24+24,j])
        DF[i,j]=np.sum(FF[i*24:i*24+24,j])

# calculate the daily averaged data
DF1=DF[:,0]/(24-DF[:,2]+10**-9)
DF2=DF[:,1]/(24-DF[:,3]+10**-9)
for i in range(len(DF1)):
    if DF1[i]>=10**9:
        DF1[i]=0
    if DF2[i]>=10**9:
        DF2[i]=0

fig=plt.figure()
plt.subplot(2,1,1)
plt.plot(np.log10(DF1+1),'b')
plt.title('Daily HighFlux')
plt.subplot(2,1,2)
plt.plot(np.log10(DF2+1),'r')
plt.show()


### linear data interpolation
## attention!!!   2013 year May data are totally missing, the first several days data of year 2011 are missing, they can't be applied interpolation;
###------------------- insert the 4 missing day data descibed in "data information" (0)
print('DF1 Shape:',DF1.shape)
print('DF2 Shape:',DF2.shape)
DF2=np.insert(DF2,1429,0)
DF1=np.insert(DF1,1429,0)
DF2=np.insert(DF2,1553,0)
DF1=np.insert(DF1,1553,0)
DF2=np.insert(DF2,1865,0)
DF1=np.insert(DF1,1865,0)
DF2=np.insert(DF2,2038,0)
DF1=np.insert(DF1,2038,0)

# find the positions (time) of the missing data
i=0
sum=0
Flag = 0
while i <len(DF2):
    if Flag==1:
        sum=0
    if DF2[i]==0:
        sum=sum+1
        Flag=0
        print(i+1,sum)
    else:
        Flag=1
    i=i+1
# result
# 1 1
# 2 2
# 3 3
# 4 4
# 5 5
# 6 6
# 298 1
# 299 2
# 300 3
# 301 4
# 302 5
# 303 6
# 304 7
# 329 1
# 330 2
# 331 3
# 332 4
# 333 5
# 334 6
# 356 1
# 357 2
# 358 3
# 359 4
# 360 5
# 361 6
# 362 7
# 363 8
# 364 9
# 365 10
# 366 11
# 367 12
# 368 13
# 1059 1
# 1060 2
# 1061 3
# 1062 4
# 1063 5
# 1064 6
# 1144 1
# 1145 2
# 1146 3
# 1147 4
# 1148 5
# 1221 1
# 1222 2
# 1223 3
# 1224 4
# 1225 5
# 1319 1
# 1320 2
# 1321 3
# 1322 4
# 1323 5
# special positions, too long time of missing data, need to be deleted
for i in range(6):
    DF2[i]=-0.1
for i in range(7):
    DF2[297+i]=-0.1
for i in range(6):
    DF2[328+i]=-0.1
for i in range(13):
    DF2[355+i]=-0.1
for i in range(6):
    DF2[1058+i]=-0.1
for i in range(5):
    DF2[1143+i]=-0.1
    DF2[1220+i]=-0.1
    DF2[1318+i]=-0.1



miss1, miss2 = 0, 0
for i in range(len(DF1)):
    if DF1[i]<=0:
        miss1 += 1
    if DF2[i]<=0:
        miss2 += 1
print('miss information:', len(DF1), len(DF2), miss1, miss2)

# plot the electron flux
fig2=plt.figure(figsize=(20, 12))
plt.subplot(2,1,1)
plt.plot(np.log10(DF1+1),'b')
plt.title('total data: '+str(len(DF2))+'   missing data: '+str(miss1), fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('$log{_1}{_0}$(flux)', fontsize=20)
plt.xlim([0, 2550])
plt.legend(['>0.8 MeV'], fontsize=20)

plt.subplot(2,1,2)
plt.plot(np.log10(DF2+1),'r')
plt.title('total data: '+str(len(DF1))+'   missing data: '+str(miss2), fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('days', fontsize=20)
plt.ylabel('$log{_1}{_0}$(flux)', fontsize=20)
plt.xlim([0, 2550])
plt.legend(['>2 MeV'], fontsize=20, loc='upper left')
plt.show()

Fdata=np.vstack((DF1,DF2)).T

savefile='HighFFModified.csv'
data = pd.DataFrame(Fdata)
data.to_csv(path + savefile, index=False, header=None, sep=',')

# # linear interpolation
# for k in range(len(DF1)):
#     if DF1[k]==0:
#         Ajust=0
#         jj=k
#         while DF1[jj]==0:
#             Ajust=Ajust+1
#             jj=jj+1
#         for kk in range(Ajust):
#             DF1[k+kk]=DF1[k-1]+(kk+1)*(DF1[k+Ajust]-DF1[k-1])/(Ajust+1)
#
# for k in range(len(DF2)):
#     if DF2[k]==0:
#         Ajust=0
#         jj=k
#         while DF2[jj]==0:
#             Ajust=Ajust+1
#             jj=jj+1
#         for kk in range(Ajust):
#             DF2[k+kk]=DF2[k-1]+(kk+1)*(DF2[k+Ajust]-DF2[k-1])/(Ajust+1)



