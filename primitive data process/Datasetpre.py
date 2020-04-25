# make daily averaged data sets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='./primitive data/'
Flux_HE='HighFFModified.csv'
Flux_LE='LowFluxDaily.csv'
SolarWind='DailySG.csv'

FHE=pd.read_table(path+Flux_HE, index_col=False, header=None, skiprows=0, sep=',')
FLE=pd.read_table(path+Flux_LE, index_col=False, header=None, skiprows=0, sep=',')
FSG=pd.read_table(path+SolarWind, index_col=False, header=None, skiprows=0, sep=',')
FHE=np.array(FHE)
FLE=np.array(FLE)
FSG=np.array(FSG)
print('HighFlux:',FHE.shape)
print('LowFlux:',FLE.shape)
print('SolarWind:',FSG.shape)


# HighFlux: (2509, 2)        miss 31+17 day data
# # LowFlux: (2522, 5)       miss 18+17 day data
# # SG: (2557, 21)    no missing data

## step1, delete 40 day data, from 2013 May 1th to 2013 June 09th;  05 01--->852th day，Index851
FHE=np.delete(FHE,np.arange(9)+851, axis=0)
FLE=np.delete(FLE,np.arange(22)+851, axis=0)
FSG=np.delete(FSG,np.arange(40)+851, axis=0)
FSG=np.delete(FSG,len(FSG)-np.arange(17)-1,axis=0)
print('FHE Shape:',FHE.shape)
print('FLE Shape:',FLE.shape)
print('FSG Shape:',FSG.shape)

# find the positions of the bad points
i=0
sum=0
Flag = 0
DF2=FHE[:,1]
while i <len(DF2):
    if Flag==1:
        sum=0
    if DF2[i]<=0:
        sum=sum+1
        Flag=0
        print(i+1,sum)
    else:
        Flag=1
    i=i+1
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
# 1050 1
# 1051 2
# 1052 3
# 1053 4
# 1054 5
# 1055 6
# 1135 1
# 1136 2
# 1137 3
# 1138 4
# 1139 5
# 1212 1
# 1213 2
# 1214 3
# 1215 4
# 1216 5
# 1310 1
# 1311 2
# 1312 3
# 1313 4
# 1314 5
## step 2, delete the first 6 days data(as they can't be intepolated) and 71 days data(many bad points)
DEL=np.hstack((np.arange(6),np.arange(71)+297))
FHE=np.delete(FHE,DEL,axis=0)
FLE=np.delete(FLE,DEL,axis=0)
FSG=np.delete(FSG,DEL,axis=0)
print("FHE shape",FHE.shape)
print('FLE shape', FLE.shape)
print('FSG shape', FSG.shape)

# calculate the number of bad points
def calzero2(data):
    numzero = 0
    for i in range(len(data)):
        if data[i] <= 0:
            numzero += 1
    return numzero

miss1 = np.zeros([2])  # [4, 141]
for i in range(2):
    miss1[i] = calzero2(FHE[:, i])


Ntime=len(FHE)
for j in range(2):
    print('!!!!:',j)
    for k in range(Ntime):
        if FHE[k,j]<=0:      ####Attention, the bad points are replaced by a special value, -0.1, some bad points are replaced by 0
            Ajust=0
            jj=k
            while FHE[jj,j]<=0:
                Ajust=Ajust+1
                jj=jj+1
            for kk in range(Ajust):
                FHE[k+kk,j]=FHE[k-1,j]+(kk+1)*(FHE[k+Ajust,j]-FHE[k-1,j])/(Ajust+1)

print('HE:',np.min(FHE))


FHE=np.log10(FHE)
FLE=np.log10(FLE)
# attention, the data should be splited into three intervals
HF1=FHE[0:291,:]
HF2=FHE[291:774,:]
HF3=FHE[774:]
LF1=FLE[0:291,:]
LF2=FLE[291:774,:]
LF3=FLE[774:]
SG1=FSG[0:291,:]
SG2=FSG[291:774,:]
SG3=FSG[774:]


print('HF3',HF3.shape)
print('LF3',LF3.shape)
print('SG3',SG3.shape)

#####-------------save data
savefile1='Daydata1.csv'
savefile2='Daydata2.csv'
savefile3='Daydata3.csv'
data = pd.DataFrame(np.hstack((HF1,LF1,SG1)))
data.to_csv(path + savefile1, index=False, header=None, sep=',')
data = pd.DataFrame(np.hstack((HF2, LF2, SG2)))
data.to_csv(path + savefile2, index=False, header=None, sep=',')
data = pd.DataFrame(np.hstack((HF3, LF3, SG3)))
data.to_csv(path + savefile3, index=False, header=None, sep=',')

# make data sets
Nday=5 ### pre 5 days
S1set1 = []  #### > 2MeV
S1set2 = []  #### medium energy electron flux
S1set3 = []  ### high and medium energy electron flux
S1set4 = []  ####Kp Dst ap AE
S1set5 = []  ####Bt Bx By Bz T N V P E

S1label1 = []  #### 1 day ahead
for i in range(len(HF1) - Nday):
    S1label1.append(HF1[i + Nday, 1])
    S1set1.append(HF1[i:i + Nday, 1])
    S1set2.append(LF1[i:i + Nday, :])
    S1set3.append(np.hstack((HF1[i:i + Nday, :], LF1[i:i + Nday, :])))
    S1set4.append(SG1[i:i + Nday, 17:21])
    S1set5.append(np.hstack(
        (SG1[i:i + Nday, 1].reshape(-1, 1), SG1[i:i + Nday, 4].reshape(-1, 1), SG1[i:i + Nday, 7:12], SG1[i:i + Nday, 14:16])))
S1set1 = np.array(S1set1)
S1set2 = np.array(S1set2)
S1set3 = np.array(S1set3)
S1set4 = np.array(S1set4)
S1set5 = np.array(S1set5)
S1label1 = np.array(S1label1)

S2set1 = []  #### > 2MeV
S2set2 = []  #### medium energy electron flux
S2set3 = []  ### high and medium energy electron flux
S2set4 = []  ####Kp Dst ap AE
S2set5 = []  ####Bt Bx By Bz T N V P E

S2label1 = []  #### 1 day ahead
for i in range(len(HF2) - Nday):
    S2label1.append(HF2[i + Nday, 1])
    S2set1.append(HF2[i:i + Nday, 1])
    S2set2.append(LF2[i:i + Nday, :])
    S2set3.append(np.hstack((HF2[i:i + Nday, :], LF2[i:i + Nday, :])))
    S2set4.append(SG2[i:i + Nday, 17:21])
    S2set5.append(np.hstack(
        (SG2[i:i + Nday, 1].reshape(-1, 1), SG2[i:i + Nday, 4].reshape(-1, 1), SG2[i:i + Nday, 7:12], SG2[i:i + Nday, 14:16])))
S2set1 = np.array(S2set1)
S2set2 = np.array(S2set2)
S2set3 = np.array(S2set3)
S2set4 = np.array(S2set4)
S2set5 = np.array(S2set5)
S2label1 = np.array(S2label1)

S3set1 = []  #### > 2MeV
S3set2 = []  #### medium energy electron flux
S3set3 = []  ### high and medium energy electron flux
S3set4 = []  ####Kp Dst ap AE
S3set5 = []  ####Bt Bx By Bz T N V P E

S3label1 = []  ### 1 day ahead
for i in range(len(HF3) - Nday):
    S3label1.append(HF3[i + Nday, 1])
    S3set1.append(HF3[i:i + Nday, 1])
    S3set2.append(LF3[i:i + Nday, :])
    S3set3.append(np.hstack((HF3[i:i + Nday, :], LF3[i:i + Nday, :])))
    S3set4.append(SG3[i:i + Nday, 17:21])
    S3set5.append(np.hstack(
        (SG3[i:i + Nday, 1].reshape(-1, 1), SG3[i:i + Nday, 4].reshape(-1, 1), SG3[i:i + Nday, 7:12], SG3[i:i + Nday, 14:16])))
S3set1 = np.array(S3set1)
S3set2 = np.array(S3set2)
S3set3 = np.array(S3set3)
S3set4 = np.array(S3set4)
S3set5 = np.array(S3set5)
S3label1 = np.array(S3label1)

Set1 = np.vstack((S1set1, S2set1, S3set1))
Set2 = np.vstack((S1set2, S2set2, S3set2))
Set3 = np.vstack((S1set3, S2set3, S3set3))
Set4 = np.vstack((S1set4, S2set4, S3set4))
Set5 = np.vstack((S1set5, S2set5, S3set5))
Label1 = np.vstack((S1label1.reshape(-1, 1), S2label1.reshape(-1, 1), S3label1.reshape(-1, 1)))

print('Set1:', Set1.shape)   # >2 MeV
print('Set3:', Set3.shape)   # 7 channels ennergy
print('Set4:', Set4.shape)   # geomagnetic indices
print('Set5:', Set5.shape)   # solar wind parameters
print('Label1:', Label1.shape)  # > 2MeV label

path2 = '../predata/'
set1file = 'Set1.csv'
set3file = 'Set3.csv'
set4file = 'Set4.csv'
set5file = 'Set5.csv'
label1file = 'Label1.csv'

data = pd.DataFrame(Set1)
data.to_csv(path2 + set1file, index=False, header=None, sep=',')
data = pd.DataFrame(Set3.reshape(-1, 35))
data.to_csv(path2 + set3file, index=False, header=None, sep=',')
data = pd.DataFrame(Set4.reshape(-1, 20))
data.to_csv(path2 + set4file, index=False, header=None, sep=',')
data = pd.DataFrame(Set5.reshape(-1, 45))
data.to_csv(path2 + set5file, index=False, header=None, sep=',')

data = pd.DataFrame(Label1)
data.to_csv(path2 + label1file, index=False, header=None, sep=',')
print(len(data))
print(len(FHE))
print('finished!')