import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# primitive data downloaded from https://cdaweb.sci.gsfc.nasa.gov
path = './primitive data/'
High_1 = 'GOES13_2011.txt'
High_2 = 'GOES13_2012.txt'
High_3 = 'GOES13_2013.txt'
High_4 = 'GOES13_2014.txt'
High_5 = 'GOES13_2015.txt'
High_6 = 'GOES13_2016.txt'
High_7 = 'GOES13_2017.txt'

# 2011 year
Flux13_1 = pd.read_table(path+High_1, header=None, skiprows=68, nrows=525600, sep='\s+')   #### no missing data
print(Flux13_1[0][0])
print(Flux13_1[1][0])
print(Flux13_1[0][len(Flux13_1)-1])
print(Flux13_1[1][len(Flux13_1)-1])
# 2012 year
Flux13_2 = pd.read_table(path+High_2, header=None, skiprows=68, nrows=527040, sep='\s+')   #### no missing data
print(Flux13_2[0][0])
print(Flux13_2[1][0])
print(Flux13_2[0][len(Flux13_2)-1])
print(Flux13_2[1][len(Flux13_2)-1])
# 2013 year
Flux13_3 = pd.read_table(path+High_3, header=None, skiprows=68, nrows=481028-68, sep='\s+')  #### miss 31 days data (data of May)
print(Flux13_3[0][0])
print(Flux13_3[1][0])
print(Flux13_3[0][len(Flux13_3)-1])
print(Flux13_3[1][len(Flux13_3)-1])
# 2014 year
Flux13_4 = pd.read_table(path+High_4, header=None, skiprows=68, nrows=524228-68, sep='\s+')   ### miss 1 day data (last day of the year)
print(Flux13_4[0][0])
print(Flux13_4[1][0])
print(Flux13_4[0][len(Flux13_4)-1])
print(Flux13_4[1][len(Flux13_4)-1])
# 2015 year
Flux13_5 = pd.read_table(path+High_5, header=None, skiprows=68, nrows=524228-68, sep='\s+')   #### miss 1 day data
print(Flux13_5[0][0])
print(Flux13_5[1][0])
print(Flux13_5[0][len(Flux13_5)-1])
print(Flux13_5[1][len(Flux13_5)-1])
# 2016 year
Flux13_6 = pd.read_table(path+High_6, header=None, skiprows=68, nrows=524228-68, sep='\s+')  #### miss 2 day data
print(Flux13_6[0][0])
print(Flux13_6[1][0])
print(Flux13_6[0][len(Flux13_6)-1])
print(Flux13_6[1][len(Flux13_6)-1])
# 2017 year
Flux13_7 = pd.read_table(path+High_7, header=None, skiprows=68, nrows=501188-68, sep='\s+')  #### miss 17 day data(last 17 day of the year)
print(Flux13_7[0][0])
print(Flux13_7[1][0])
print(Flux13_7[0][len(Flux13_7)-1])
print(Flux13_7[1][len(Flux13_7)-1])

###------ find the position of the missing data
month = np.zeros([12])
for i in range(len(Flux13_3)):
    num = Flux13_3[0][i].split('-')[1]
    for j in range(0, 12):
        if int(num) == j+1:
            month[j] = month[j]+1
print('2013 Month:', month/60/24)   #result: 2013 Month: [31. 28. 31. 30.  0. 30. 31. 31. 30. 31. 30. 31.]    ### miss 31 day data (May)

month = np.zeros([12])
for i in range(len(Flux13_4)):
    num = Flux13_4[0][i].split('-')[1]
    for j in range(0, 12):
        if int(num) == j+1:
            month[j] = month[j]+1
print('2014 Month:', month/60/24)   #result: 2014 Month: [31. 28. 31. 30. 31. 30. 31. 31. 30. 31. 30. 30.]    #### miss 1 day data (the last day)

month = np.zeros([12])
for i in range(len(Flux13_5)):
    num = Flux13_5[0][i].split('-')[1]
    for j in range(0, 12):
        if int(num) == j+1:
            month[j] = month[j]+1
print('2015 Month:', month/60/24)  #result: 2015 Month: [31. 28. 31. 30. 30. 30. 31. 31. 30. 31. 30. 31.]   #### miss 1 day data (the date is May 4th)

for i in range(30):
    print(Flux13_5[0][172800+i*1440])
    print(Flux13_5[0][172800+i*1440+1439])

month = np.zeros([12])
for i in range(len(Flux13_6)):
    num = Flux13_6[0][i].split('-')[1]
    for j in range(0, 12):
        if int(num) == j+1:
            month[j] = month[j]+1
print('2016 Month:', month/60/24)   #result: 2016 Month: [31. 29. 30. 30. 31. 30. 31. 30. 30. 31. 30. 31.]   #### miss 2 day data  ## The dates are March 11th and August 31th

for i in range(30):
    print(Flux13_6[0][86400+i*1440])
    print(Flux13_6[0][86400+i*1440+1439])

for i in range(30):
    print(Flux13_6[0][305280+i*1440])
    print(Flux13_6[0][305280+i*1440+1439])

print('2011', Flux13_1.shape)
print('2012', Flux13_2.shape)
print('2013', Flux13_3.shape)
print('2014', Flux13_4.shape)
print('2015', Flux13_5.shape)
print('2016', Flux13_6.shape)
print('2017', Flux13_7.shape)
####------------------------------------------
Flux13_1 = np.array(Flux13_1)
Flux13_2 = np.array(Flux13_2)
Flux13_3 = np.array(Flux13_3)
Flux13_4 = np.array(Flux13_4)
Flux13_5 = np.array(Flux13_5)
Flux13_6 = np.array(Flux13_6)
Flux13_7 = np.array(Flux13_7)
print('New Shape', Flux13_1.shape)

N1 = len(Flux13_1)
N2 = len(Flux13_2)
N3 = len(Flux13_3)
N4 = len(Flux13_4)
N5 = len(Flux13_5)
N6 = len(Flux13_6)
N7 = len(Flux13_7)


# calculate how many bad data
S1 = np.zeros([7, 4])
print('S1', S1)
print('N1', N1)
for i in range(N1):
    for j in range(4):
        if Flux13_1[i, 2+j] <= 0:
            Flux13_1[i, 2+j] = 0
            S1[0, j] += 1
print('2011', N1, S1[0])           # result: 525600, [  3304.   3213. 196403. 218759.]

for i in range(N2):
    for j in range(4):
        if Flux13_2[i, 2+j] <= 0:
            Flux13_2[i, 2+j] = 0
            S1[1, j] = S1[1, j]+1
print('2012', N2, S1[1])            # result: 527040,[  1830.   1791. 187517. 212420.]

for i in range(N3):
    for j in range(4):
        if Flux13_3[i, 2+j] <= 0:
            Flux13_3[i, 2+j] = 0
            S1[2, j] = S1[2, j]+1
print('2013', N3, S1[2])             # result: 480960,[  2563.   2547. 187975. 202833.]

for i in range(N4):
    for j in range(4):
        if Flux13_4[i, 2+j] <= 0:
            Flux13_4[i, 2+j] = 0
            S1[3, j] = S1[3, j]+1
print('2014', N4, S1[3])             # result: 524160,[  5228.   5134. 302253. 322735.]

for i in range(N5):
    for j in range(4):
        if Flux13_5[i, 2+j] <= 0:
            Flux13_5[i, 2+j] = 0
            S1[4, j] = S1[4, j]+1
print('2015', N5, S1[4])             # result: 524160,[  5215.   5015. 158924. 172566.]

for i in range(N6):
    for j in range(4):
        if Flux13_6[i, 2+j] <= 0:
            Flux13_6[i, 2+j] = 0
            S1[5, j] = S1[5, j]+1
print('2016', N6, S1[5])             # result: 524160,[  1505.   1484. 124376. 134191.]

for i in range(N7):
    for j in range(4):
        if Flux13_7[i, 2+j] <= 0:
            Flux13_7[i, 2+j] = 0
            S1[6, j] = S1[6, j]+1
print('2017', N7, S1[6])             # result: 501120,[  4513.   4289.  99076. 110878.]

#####====================
## data infomation
#2011 no missing data
#2012 no missing data
#2013 miss 31 data (May)
#2014 miss 1 day data (Dec. 31th)
#2015 miss 1 day data, May 4th
#2016 miss 2 day data，March 11th and August 31th
#2017 miss 17 day data, from Dec. 15th to Dec. 31th

####----------------------------------------------------
# calculate the hourly averaged data
def calzero(data):
    numzero = 0
    for i in range(60):
        if data[i] == 0:
            numzero += 1
    return numzero

### the first 4 columns are used to calculate sum, and the last 4 columns are used to count zeros
F1 = np.zeros([int(N1/60), 8])
for i in range(int(N1/60)):
    for j in range(4):
        F1[i, j+4] = calzero(Flux13_1[i*60: i*60+60, 2+j])
        F1[i, j] = np.sum(Flux13_1[i*60: i*60+60, 2+j])

F2 = np.zeros([int(N2/60), 8])
for i in range(int(N2/60)):
    for j in range(4):
        F2[i, j+4] = calzero(Flux13_2[i*60: i*60+60, 2+j])
        F2[i, j] = np.sum(Flux13_2[i*60: i*60+60, 2+j])

F3 = np.zeros([int(N3/60), 8])
for i in range(int(N3/60)):
    for j in range(4):
        F3[i, j+4] = calzero(Flux13_3[i*60: i*60+60, 2+j])
        F3[i, j] = np.sum(Flux13_3[i*60: i*60+60, 2+j])

F4 = np.zeros([int(N4/60), 8])
for i in range(int(N4/60)):
    for j in range(4):
        F4[i, j+4] = calzero(Flux13_4[i*60: i*60+60, 2+j])
        F4[i, j] = np.sum(Flux13_4[i*60: i*60+60, 2+j])

F5 = np.zeros([int(N5/60), 8])
for i in range(int(N5/60)):
    for j in range(4):
        F5[i, j+4] = calzero(Flux13_5[i*60: i*60+60, 2+j])
        F5[i, j] = np.sum(Flux13_5[i*60: i*60+60, 2+j])

F6 = np.zeros([int(N6/60), 8])
for i in range(int(N6/60)):
    for j in range(4):
        F6[i, j+4] = calzero(Flux13_6[i*60: i*60+60, 2+j])
        F6[i, j] = np.sum(Flux13_6[i*60: i*60+60, 2+j])

F7 = np.zeros([int(N7/60), 8])
for i in range(int(N7/60)):
    for j in range(4):
        F7[i, j+4] = calzero(Flux13_7[i*60: i*60+60, 2+j])
        F7[i, j] = np.sum(Flux13_7[i*60: i*60+60, 2+j])

#####====================
# names = locals()
# for i in range(1, 8):
#     names['savefile'+str(i)] = 'HighFlux'+str(i)+'.csv'
#     data = pd.DataFrame(names['F'+str(i)])
#     data.to_csv(path + names['savefile'+str(i)], index=False, header=None, sep=',')


### check the missing data information of >0.8MeV
# >0.8MeV data miss few data (nearly perfect)
sum = 0
for i in range(len(F1)):
    if F1[i, 5] >= 58 or F1[i, 4] >= 58:
        sum += 1
print('sum1', sum)   ##24
sum = 0
for i in range(len(F2)):
    if F2[i, 5] >= 58 or F2[i, 4]>=58:
        sum += 1
print('sum2', sum)   ##0
sum = 0
for i in range(len(F3)):
    if F3[i, 5] >= 58 or F3[i, 4] >= 58:
        sum += 1
print('sum3', sum)   ##24
sum = 0
for i in range(len(F4)):
    if F4[i, 5] >= 58 or F4[i, 4] >= 58:
        sum += 1
print('sum4', sum)   ##40
sum = 0
for i in range(len(F5)):
    if F5[i, 5] >= 58 or F5[i, 4] >= 58:
        sum += 1
print('sum5', sum)   ##43
sum = 0
for i in range(len(F6)):
    if F6[i, 5] >= 58 or F6[i, 4]>= 58:
        sum += 1
print('sum6', sum)   ##3
for i in range(len(F7)):
    sum = 0
    if F7[i, 5] >= 58 or F7[i, 4]>=58:
        sum += 1
print('sum7', sum)   ##1

# check the missing data information of >2MeV
sum = 0
for i in range(len(F1)):
    if F1[i, 6] >= 60 and F1[i, 7] >= 60:
        sum += 1
print('sum1', sum)   ##2318
sum = 0
for i in range(len(F2)):
    if F2[i, 6] >= 60 and F2[i, 7] >= 60:
        sum += 1
print('sum2', sum)   ##1927
sum = 0
for i in range(len(F3)):
    if F3[i, 6] >= 60 and F3[i, 7] >= 60:
        sum += 1
print('sum3', sum)   ##2217
sum = 0
for i in range(len(F4)):
    if F4[i, 6] >= 60 and F4[i, 7] >= 60:
        sum += 1
print('sum4', sum)   ##3893
sum = 0
for i in range(len(F5)):
    if F5[i, 6] >= 60 and F5[i, 7] >= 60:
        sum += 1
print('sum5', sum)   ##1606
sum = 0
for i in range(len(F6)):
    if F6[i, 6] >= 60 and F6[i, 7] >= 60:
        sum += 1
print('sum6', sum)   ##1345
sum = 0
for i in range(len(F7)):
    if F7[i, 6] >= 60 and F7[i, 7] >= 60:
        sum += 1
print('sum7', sum)   ##1

### calculate the hourly averaged data
####---------------------------------------F1
NF11=F1[:,0]/(60-F1[:,4]+10**-8)
NF12=F1[:,1]/(60-F1[:,5]+10**-8)
NF13=F1[:,2]/(60-F1[:,6]+10**-8)
NF14=F1[:,3]/(60-F1[:,7]+10**-8)

for i in range(len(F1)):
    if NF11[i]>2*10**7:
        NF11[i]=0
    if NF12[i]>2*10**7:
        NF12[i]=0
    if NF13[i]>2*10**7:
        NF13[i]=0
    if NF14[i]>2*10**7:
        NF14[i]=0

for i in range(len(NF11)):            #### A B detectors calibration
    if NF11[i]==0 and NF12[i]!=0:
        NF11[i]=NF12[i]
    if NF11[i]!=0 and NF12[i]==0:
        NF12[i]=NF11[i]
    if NF13[i]==0 and NF14[i]!=0:
        NF13[i]=NF14[i]
    if NF13[i]!=0 and NF14[i]==0:
        NF14[i]=NF13[i]

fig1=plt.figure()
plt.subplot(2,1,1)
plt.plot(np.log10(NF11+NF12+1),'b',linewidth=0.5)
plt.title('F1')
plt.subplot(2,1,2)
plt.plot(np.log10(NF13+NF14+1),'b',linewidth=0.5)
plt.show()

FF1=np.vstack((NF11+NF12,NF13+NF14)).T
####---------------------------------------F2
NF11=F2[:,0]/(60-F2[:,4]+10**-8)
NF12=F2[:,1]/(60-F2[:,5]+10**-8)
NF13=F2[:,2]/(60-F2[:,6]+10**-8)
NF14=F2[:,3]/(60-F2[:,7]+10**-8)

for i in range(len(F2)):
    if NF11[i]>2*10**7:
        NF11[i]=0
    if NF12[i]>2*10**7:
        NF12[i]=0
    if NF13[i]>2*10**7:
        NF13[i]=0
    if NF14[i]>2*10**7:
        NF14[i]=0

for i in range(len(NF11)):            #### A B detectors calibration
    if NF11[i]==0 and NF12[i]!=0:
        NF11[i]=NF12[i]
    if NF11[i]!=0 and NF12[i]==0:
        NF12[i]=NF11[i]
    if NF13[i]==0 and NF14[i]!=0:
        NF13[i]=NF14[i]
    if NF13[i]!=0 and NF14[i]==0:
        NF14[i]=NF13[i]

fig1=plt.figure()
plt.subplot(2,1,1)
plt.plot(np.log10(NF11+NF12+1),'b',linewidth=0.5)
plt.title('F2')
plt.subplot(2,1,2)
plt.plot(np.log10(NF13+NF14+1),'b',linewidth=0.5)
plt.show()

FF2=np.vstack((NF11+NF12,NF13+NF14)).T
####---------------------------------------F3
NF11=F3[:,0]/(60-F3[:,4]+10**-8)
NF12=F3[:,1]/(60-F3[:,5]+10**-8)
NF13=F3[:,2]/(60-F3[:,6]+10**-8)
NF14=F3[:,3]/(60-F3[:,7]+10**-8)

for i in range(len(F3)):
    if NF11[i]>2*10**7:
        NF11[i]=0
    if NF12[i]>2*10**7:
        NF12[i]=0
    if NF13[i]>2*10**7:
        NF13[i]=0
    if NF14[i]>2*10**7:
        NF14[i]=0

for i in range(len(NF11)):            #### A B detectors calibration
    if NF11[i]==0 and NF12[i]!=0:
        NF11[i]=NF12[i]
    if NF11[i]!=0 and NF12[i]==0:
        NF12[i]=NF11[i]
    if NF13[i]==0 and NF14[i]!=0:
        NF13[i]=NF14[i]
    if NF13[i]!=0 and NF14[i]==0:
        NF14[i]=NF13[i]

fig1=plt.figure()
plt.subplot(2,1,1)
plt.plot(np.log10(NF11+NF12+1),'b',linewidth=0.5)
plt.title('F3')
plt.subplot(2,1,2)
plt.plot(np.log10(NF13+NF14+1),'b',linewidth=0.5)
plt.show()

FF3=np.vstack((NF11+NF12,NF13+NF14)).T
####---------------------------------------F4
NF11=F4[:,0]/(60-F4[:,4]+10**-8)
NF12=F4[:,1]/(60-F4[:,5]+10**-8)
NF13=F4[:,2]/(60-F4[:,6]+10**-8)
NF14=F4[:,3]/(60-F4[:,7]+10**-8)

for i in range(len(F4)):
    if NF11[i]>2*10**7:
        NF11[i]=0
    if NF12[i]>2*10**7:
        NF12[i]=0
    if NF13[i]>2*10**7:
        NF13[i]=0
    if NF14[i]>2*10**7:
        NF14[i]=0

for i in range(len(NF11)):            #### A B detectors calibration
    if NF11[i]==0 and NF12[i]!=0:
        NF11[i]=NF12[i]
    if NF11[i]!=0 and NF12[i]==0:
        NF12[i]=NF11[i]
    if NF13[i]==0 and NF14[i]!=0:
        NF13[i]=NF14[i]
    if NF13[i]!=0 and NF14[i]==0:
        NF14[i]=NF13[i]

fig1=plt.figure()
plt.subplot(2,1,1)
plt.plot(np.log10(NF11+NF12+1),'b',linewidth=0.5)
plt.title('F4')
plt.subplot(2,1,2)
plt.plot(np.log10(NF13+NF14+1),'b',linewidth=0.5)
plt.show()

FF4=np.vstack((NF11+NF12,NF13+NF14)).T
####---------------------------------------F5
NF11=F5[:,0]/(60-F5[:,4]+10**-8)
NF12=F5[:,1]/(60-F5[:,5]+10**-8)
NF13=F5[:,2]/(60-F5[:,6]+10**-8)
NF14=F5[:,3]/(60-F5[:,7]+10**-8)

for i in range(len(F5)):
    if NF11[i]>2*10**7:
        NF11[i]=0
    if NF12[i]>2*10**7:
        NF12[i]=0
    if NF13[i]>2*10**7:
        NF13[i]=0
    if NF14[i]>2*10**7:
        NF14[i]=0

for i in range(len(NF11)):            #### A B detectors calibration
    if NF11[i]==0 and NF12[i]!=0:
        NF11[i]=NF12[i]
    if NF11[i]!=0 and NF12[i]==0:
        NF12[i]=NF11[i]
    if NF13[i]==0 and NF14[i]!=0:
        NF13[i]=NF14[i]
    if NF13[i]!=0 and NF14[i]==0:
        NF14[i]=NF13[i]

fig1=plt.figure()
plt.subplot(2,1,1)
plt.plot(np.log10(NF11+NF12+1),'b',linewidth=0.5)
plt.title('F5')
plt.subplot(2,1,2)
plt.plot(np.log10(NF13+NF14+1),'b',linewidth=0.5)
plt.show()

FF5=np.vstack((NF11+NF12,NF13+NF14)).T
####---------------------------------------F6
NF11=F6[:,0]/(60-F6[:,4]+10**-8)
NF12=F6[:,1]/(60-F6[:,5]+10**-8)
NF13=F6[:,2]/(60-F6[:,6]+10**-8)
NF14=F6[:,3]/(60-F6[:,7]+10**-8)

for i in range(len(F6)):
    if NF11[i]>2*10**7:
        NF11[i]=0
    if NF12[i]>2*10**7:
        NF12[i]=0
    if NF13[i]>2*10**7:
        NF13[i]=0
    if NF14[i]>2*10**7:
        NF14[i]=0

for i in range(len(NF11)):            #### A B detectors calibration
    if NF11[i]==0 and NF12[i]!=0:
        NF11[i]=NF12[i]
    if NF11[i]!=0 and NF12[i]==0:
        NF12[i]=NF11[i]
    if NF13[i]==0 and NF14[i]!=0:
        NF13[i]=NF14[i]
    if NF13[i]!=0 and NF14[i]==0:
        NF14[i]=NF13[i]

fig1=plt.figure()
plt.subplot(2,1,1)
plt.plot(np.log10(NF11+NF12+1),'b',linewidth=0.5)
plt.title('F6')
plt.subplot(2,1,2)
plt.plot(np.log10(NF13+NF14+1),'b',linewidth=0.5)
plt.show()

FF6=np.vstack((NF11+NF12,NF13+NF14)).T
####---------------------------------------F7
NF11=F7[:,0]/(60-F7[:,4]+10**-8)
NF12=F7[:,1]/(60-F7[:,5]+10**-8)
NF13=F7[:,2]/(60-F7[:,6]+10**-8)
NF14=F7[:,3]/(60-F7[:,7]+10**-8)

for i in range(len(F7)):
    if NF11[i]>2*10**7:
        NF11[i]=0
    if NF12[i]>2*10**7:
        NF12[i]=0
    if NF13[i]>2*10**7:
        NF13[i]=0
    if NF14[i]>2*10**7:
        NF14[i]=0

for i in range(len(NF11)):            #### A B detectors calibration
    if NF11[i]==0 and NF12[i]!=0:
        NF11[i]=NF12[i]
    if NF11[i]!=0 and NF12[i]==0:
        NF12[i]=NF11[i]
    if NF13[i]==0 and NF14[i]!=0:
        NF13[i]=NF14[i]
    if NF13[i]!=0 and NF14[i]==0:
        NF14[i]=NF13[i]

fig1=plt.figure()
plt.subplot(2,1,1)
plt.plot(np.log10(NF11+NF12+1),'b',linewidth=0.5)
plt.title('F7')
plt.subplot(2,1,2)
plt.plot(np.log10(NF13+NF14+1),'b',linewidth=0.5)
plt.show()

FF7=np.vstack((NF11+NF12,NF13+NF14)).T

FF=np.vstack((FF1,FF2,FF3,FF4,FF5,FF6,FF7))
print('FF Shape:', FF.shape)  ####FF Shape: (60120, 8)   2505 days data

savefile='HighFF.csv'
data = pd.DataFrame(FF)
data.to_csv(path + savefile, index=False, header=None, sep=',')






