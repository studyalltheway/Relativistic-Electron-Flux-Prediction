import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='./primitive data/'
datafile1='OMNIWeb Results.html'
savedatafile='DailySG.csv'
####-----------------------------------------
#  1 Scalar B, nT
#  2 Vector B Magnitude,nT
#  3 Lat. Angle of B (GSE)
#  4 Long. Angle of B (GSE)
#  5 BX, nT (GSE, GSM)
#  6 BY, nT (GSE)
#  7 BZ, nT (GSE)
#  8 BY, nT (GSM)
#  9 BZ, nT (GSM)
#### 10 SW Plasma Temperature, K     ！！87 bad points
#### 11 SW Proton Density, N/cm^3    ！！！552 bad points
# 12 SW Plasma Speed, km/s
# 13 SW Plasma flow long. angle
# 14 SW Plasma flow lat. angle
#### 15 Flow pressure                ！！！552 bad points
# 16 E elecrtric field
#### 17 Plasma betta                 ！！！557 bad points
# 18 Kp index
# 19 Dst-index, nT
# 20 ap_index, nT
# 21 AE-index, nT

SW = pd.read_table(path+datafile1, header=None, skiprows=29, nrows=61368, sep='\s+')
SW=np.array(SW)
print('SW shape:', SW.shape)
print(SW[0][0])
print(SW[0][1])
print(SW[0][2])
print(SW[len(SW)-1][0])
print(SW[len(SW)-1][1])
print(SW[len(SW)-1][2])

for j in range(SW.shape[1]):
    print('%d：'%(j+1),np.max(SW[:,j]),np.min(SW[:,j]),np.median(SW[:,j]))
# 1： 2017.0 2011.0 2014.0
# 2： 366.0 1.0 183.0
# 3： 23.0 0.0 11.5
# 4： 999.9 0.5 5.1
# 5： 999.9 0.2 4.5
# 6： 999.9 -89.7 -0.3
# 7： 999.9 -22.4 -0.2
# 9： 999.9 0.0 176.1
# 8： 999.9 -25.1 0.1
# 10： 999.9 -26.3 0.0
# 11： 999.9 -24.3 0.1
# 12： 999.9 -26.3 -0.1
# 13： 9999999.0 5315.0 64656.0
# 14： 999.9 0.1 4.9
# 15： 9999.0 242.0 402.0
# 16： 999.9 -23.9 -0.4
# 17： 999.9 -14.4 -0.3
# 18： 99.99 0.05 1.61
# 19： 999.99 -17.32 0.03
# 20： 999.99 0.01 1.59
# 21： 83.0 0.0 17.0
# 22： 77.0 -223.0 -8.0
# 23： 236.0 0.0 6.0
# 24： 1842.0 5.0 95.0
# check how many bad points of each variable
for i in range(SW.shape[0]):   ###2557day×24hour
    for j in range(9):
       if SW[i,j+3]>990:
           SW[i,j+3]=10**-10      ####special value, set as 10**-10
    if SW[i,12]>9.99*10**6:
        SW[i,12]=10**-10
    if SW[i,13]>9.99*10**2:
        SW[i,13]=10**-10
    if SW[i,14]>9.99*10**3:
        SW[i,14]=10**-10
    if SW[i,15]>9.99*10**2:
        SW[i,15]=10**-10
    if SW[i,16]>9.99*10**2:
        SW[i,16]=10**-10
    if SW[i,17]>9.99*10**1:
        SW[i,17]=10**-10
    if SW[i,18]>9.99*10**2:
        SW[i,18]=10**-10
    if SW[i,19]>9.99*10**2:
        SW[i,19]=10**-10
for j in range(SW.shape[1]-3):
    sum=0
    for i in range(SW.shape[0]):
        if SW[i,j+3]==10**-10:
            sum=sum+1
    print('%d：'%(j+4),sum)
# 4： 6
# 5： 6
# 6： 6
# 7： 6
# 8： 32
# 9： 32
# 10： 0
# 11： 0
# 12： 0
# 13： 87
# 14： 552
# 15： 12
# 16： 12
# 17： 12
# 18： 552
# 19： 38
# 20： 557
# 21： 0
# 22： 0
# 23： 0
# 24： 0

# NSW=np.zeros([int(SW.shape[0],18)])
# DF2=SW[:,12]
# i=0
# sum=0
# Flag = 0
# while i <len(DF2):
#     if Flag==1:
#         sum=0
#     if DF2[i]==0.0001:
#         sum=sum+1
#         Flag=0
#         print(i+1,sum)
#     else:
#         Flag=1
#     i=i+1


# calculate the daily averaged data
def calzero(data):
    numzero=0
    for i in range(24):
        if data[i]==10**-10:
            numzero += 1
    return numzero

### the first 21 columns are used to calculate sum, and the last 21 columns are used to count zeros
N=len(SW)
DF=np.zeros([int(N/24),42])
for i in range(int(N/24)):
    for j in range(21):
        DF[i,j+21]=calzero(SW[i*24:i*24+24,j+3])
        DF[i,j]=np.sum(SW[i*24:i*24+24,j+3])
print('DF---')

NDF=np.zeros([int(N/24),21])
for j in range(21):
    NDF[:,j]=DF[:,j]/(24-DF[:,21+j]+10**-6)

sum=0
for i in range(int(N/24)):
    for j in range(21):
        if NDF[i,j]<=24*10**-4 and NDF[i,j]>=24*10**-4:
           NDF[i,j]=10**-4
           sum=sum+1
print('sum',sum)                        ###sum=9
print('NDF Shape:',NDF.shape)

for i in range(3):
    print('%d:'%(i+6),np.max(NDF[:,i+6]),np.min(NDF[:,i+6]),np.median(NDF[:,i+6]))

# calculate how many bad points in daily averaged data
miss = np.zeros([21])
def calzero2(data):
    numzero=0
    for i in range(len(data)):
        if data[i]==10**-4:
            numzero += 1
    return numzero

for i in range(21):
    miss[i] = calzero2(NDF[:, i])

print('missing information:', miss)

# interpolate bad points
Ntime=len(NDF)
for j in range(21):
    print('!!!!:',j)
    for k in range(Ntime):
        if NDF[k,j]==10**-4:
            Ajust=0
            jj=k
            while NDF[jj,j]==10**-4:
                Ajust=Ajust+1
                jj=jj+1
            for kk in range(Ajust):
                NDF[k+kk,j]=NDF[k-1,j]+(kk+1)*(NDF[k+Ajust,j]-NDF[k-1,j])/(Ajust+1)

data=pd.DataFrame(NDF)
data.to_csv(path+savedatafile, index=False, header=None, sep=',')

fig=plt.figure()
for j in range(21):
   plt.subplot(7,3,j+1)
   plt.plot(NDF[:,j])
plt.show()














