import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='./primitive data/'
Low1='GOES13_LE_2011.txt'
Low2='GOES13_LE_2012.txt'
Low3='GOES13_LE_2013.txt'
Low4='GOES13_LE_2014.txt'
Low5='GOES13_LE_2015.txt'
Low6='GOES13_LE_2016.txt'
Low7='GOES13_LE_2017.txt'

Flux1=pd.read_table(path+Low1,header=None,skiprows=64,nrows=105120,sep='\s+')   #### no missing data
print(Flux1[0][0])
print(Flux1[1][0])
print(Flux1[0][len(Flux1)-1])
print(Flux1[1][len(Flux1)-1])

Flux2=pd.read_table(path+Low2,header=None,skiprows=64,nrows=105408,sep='\s+')   #### no missing data
print(Flux2[0][0])
print(Flux2[1][0])
print(Flux2[0][len(Flux2)-1])
print(Flux2[1][len(Flux2)-1])
Flux3=pd.read_table(path+Low3,header=None,skiprows=64,nrows=105120-2880*2+288*2,sep='\s+')  #### miss 18 day data
print(Flux3[0][0])
print(Flux3[1][0])
print(Flux3[0][len(Flux3)-1])
print(Flux3[1][len(Flux3)-1])
Flux4=pd.read_table(path+Low4,header=None,skiprows=64,nrows=104832,sep='\s+')   ### miss last day data
print(Flux4[0][0])
print(Flux4[1][0])
print(Flux4[0][len(Flux4)-1])
print(Flux4[1][len(Flux4)-1])
Flux5=pd.read_table(path+Low5,header=None,skiprows=64,nrows=104832,sep='\s+')   #### miss 1 day data
print(Flux5[0][0])
print(Flux5[1][0])
print(Flux5[0][len(Flux5)-1])
print(Flux5[1][len(Flux5)-1])
Flux6=pd.read_table(path+Low6,header=None,skiprows=64,nrows=104832+288,sep='\s+')  #### miss 1 day data    104832===364day
print(Flux6[0][0])
print(Flux6[1][0])
print(Flux6[0][len(Flux6)-1])
print(Flux6[1][len(Flux6)-1])
Flux7=pd.read_table(path+Low7,header=None,skiprows=64,nrows=100224,sep='\s+')  #### miss last 17 day data
print(Flux7[0][0])
print(Flux7[1][0])
print(Flux7[0][len(Flux7)-1])
print(Flux7[1][len(Flux7)-1])

###------find the positions of the missing data
month=np.zeros([12])
for i in range(len(Flux3)):
    num=Flux3[0][i].split('-')[1]
    for j in range(0,12):
        if int(num)==j+1:
            month[j]=month[j]+1
print('2013 Month:',month/12/24)   ###2013 Month: [31. 28. 31. 30. 22. 21. 31. 31. 30. 31. 30. 31.]    ###miss 18 day data  from May 23th to June 09th

month=np.zeros([12])
for i in range(len(Flux4)):
    num=Flux4[0][i].split('-')[1]
    for j in range(0,12):
        if int(num)==j+1:
            month[j]=month[j]+1
print('2014 Month:',month/12/24)   ###2014 Month: [31. 28. 31. 30. 31. 30. 31. 31. 30. 31. 30. 30.]    #### miss 1 day data (the last day)

month=np.zeros([12])
for i in range(len(Flux5)):
    num=Flux5[0][i].split('-')[1]
    for j in range(0,12):
        if int(num)==j+1:
            month[j]=month[j]+1
print('2015 Month:',month/12/24)  ####2015 Month: [31. 28. 31. 30. 30. 30. 31. 31. 30. 31. 30. 31.]   #### miss 1 day data   (May 4th)

month=np.zeros([12])
for i in range(len(Flux6)):
    num=Flux6[0][i].split('-')[1]
    for j in range(0,12):
        if int(num)==j+1:
            month[j]=month[j]+1
print('2016 Month:',month/12/24)  ####2016 Month: [31. 29. 30. 30. 31. 30. 31. 31. 30. 31. 30. 31.]   #### miss 1 day data    (March 11th)

for i in range(50):      ####2013
    print(Flux3[0][34560+i*288])
    print(Flux3[0][34560 + i * 288+287])

for i in range(30):
    print(Flux5[0][34560+i*288])
    print(Flux5[0][34560 + i * 288+287])

for i in range(30):
    print(Flux6[0][17280+i*288])
    print(Flux6[0][17280 + i * 288+287])


# calculate the hourly averaged data
Flux1=np.array(Flux1)
Flux2=np.array(Flux2)
Flux3=np.array(Flux3)
Flux4=np.array(Flux4)
Flux5=np.array(Flux5)
Flux6=np.array(Flux6)
Flux7=np.array(Flux7)
print('New Shape',Flux1.shape)

FF=np.vstack((Flux1,Flux2,Flux3,Flux4,Flux5,Flux6,Flux7))
print('FF Shape:',FF.shape)

S1=np.zeros([45])
N1=len(FF)
print('N1',N1)
for i in range(N1):
    for j in range(45):
        if FF[i,2+j]<=0:
            FF[i,2+j] = 0
            S1[j]=S1[j]+1
print('S1',S1)                  ####2976


def calzero(data):
    numzero=0
    for i in range(12):
        if data[i]==0:
            numzero+=1
    return numzero


### the first 45 columns are used to calculate sum, and the last 45 columns are used to count zeros
HF=np.zeros([int(N1/12),90])
for i in range(int(N1/12)):
    for j in range(45):
        HF[i,j+45]=calzero(FF[i*12:i*12+12,2+j])
        HF[i,j]=np.sum(FF[i*12:i*12+12,2+j])
print('HF---',HF.shape)
#####====================
savefile='LowFLux.csv'
data = pd.DataFrame(HF)
data.to_csv(path + savefile, index=False, header=None, sep=',')




