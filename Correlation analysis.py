## figure 1 of the paper,
## Correlation analysis between relativistic electron flux and input variables with different offset days.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path2='./predata'
set1file='/Set1.csv'
set3file='/Set3.csv'
set4file='/Set4.csv'
set5file='/Set5.csv'
label1file='/Label1.csv'

Set1 = pd.read_table(path2+set1file, index_col=False, header=None, skiprows=0, sep=',')
Set1=np.array(Set1)
print('Set1 shape:',Set1.shape)

Set3 = pd.read_table(path2+set3file, index_col=False, header=None, skiprows=0, sep=',')
Set3=np.array(Set3)
print('Set3 shape:',Set3.shape)
Set4 = pd.read_table(path2+set4file, index_col=False, header=None, skiprows=0, sep=',')
Set4=np.array(Set4)
print('Set4 shape:',Set4.shape)
Set5 = pd.read_table(path2+set5file, index_col=False, header=None, skiprows=0, sep=',')
Set5=np.array(Set5)
print('Set5 shape:',Set5.shape)


Label1 = pd.read_table(path2+label1file, index_col=False, header=None, skiprows=0, sep=',')
Label1=np.array(Label1)
#
print('Label1 shape:', Label1.shape)
Label1=Label1.reshape(-1)

# ture label1, unit transformation from log10 cm-2s-1sr-1 to log10 cm-2d-1sr-1
Label1 += 4.936514

###-----auto correlation coefficients of high energy electron flux
Auto=[]
N=len(Label1)
for i in range(30):
    data1=Label1[0:N-30]
    data2=Label1[i:N-30+i]
    cor=np.corrcoef(data1,data2)[0,1]
    Auto.append(cor)
print('Auto',Auto)
###------correlation coefficients between high energy electron flux and medium energy electron flux
F1=Set3[:,1]
F2=Set3[:,0]
F3=Set3[:,6]
F4=Set3[:,5]
F5=Set3[:,4]
F6=Set3[:,3]
F7=Set3[:,2]
FF=np.vstack((F1,F2,F3,F4,F5,F6,F7)).T
# unit transformation
FF += 4.936514

HLcor=[]
for i in range(7):
    Lcor=[]
    for j in range(90):
        data1=Label1[70:N-20]
        data2=FF[j:N-90+j,i]
        cor=np.corrcoef(data1,data2)[0,1]
        Lcor.append(cor)
    Lcor.reverse()
    HLcor.append(Lcor)

HLcor=np.array(HLcor)
print('HLcor:',HLcor.shape)

HGcor=[]
for i in range(20):
    Gcor=[]
    for j in range(90):
        data1=Label1[70:N-20]
        data2=Set4[j:N-90+j,i]
        cor=np.corrcoef(data1,data2)[0,1]
        Gcor.append(cor)
    Gcor.reverse()
    HGcor.append(Gcor)
HGcor=np.array(HGcor)
print('HGcor:',HGcor.shape)

HScor=[]
for i in range(9):
    Scor=[]
    for j in range(90):
        data1=Label1[70:N-20]
        data2=Set5[j:N-90+j,i]
        cor=np.corrcoef(data1,data2)[0,1]
        Scor.append(cor)
    Scor.reverse()
    HScor.append(Scor)
HScor=np.array(HScor)
print('HScor:',HScor.shape)


fig2 = plt.figure(figsize=(20, 12))
plt.subplots_adjust(left=5, bottom=5, right=95, top=95, wspace=None, hspace=0)

plt.subplot(311)
color = ['m', 'y', 'pink', 'tan', 'coral',  'c',  'b', 'g',     'r']
for i in range(9):
    # don't plot Bx, By, Bz
    if i in [1, 2, 3]: # Bx, By, Bz
        continue
    plt.plot(np.abs(HScor[i][10:30]), 'o--', color=color[i], linewidth=2, markersize=12)
plt.xlim([3.97, 13.5])
plt.ylim([-0.01, 0.8])
x = np.arange(0, 1, 0.2)
plt.yticks(x, ['0', '0.2', '0.4', '0.6', '0.8'], fontsize=20)
plt.xticks([])
plt.ylabel('|LC|', fontsize=25)
plt.legend(['   B','   T','   N','   V','   P','   Ey'], loc='upper right', fontsize=15, framealpha=1)

plt.subplot(312)
color=['b', 'g', 'y', 'c']
for i in range(4):
    plt.plot(np.abs(HGcor[i][10:30]), 'o--', linewidth=2, markersize=12)#,color=color[i])
plt.xlim([3.97, 13.5])
plt.ylim([-0.01, 0.8])
plt.xticks([])
x = np.arange(0, 0.8, 0.2)
plt.yticks(x, ['0', '0.2', '0.4', '0.6', '0.8'], fontsize=20)
plt.ylabel('|LC|', fontsize=25)
plt.legend([' Kp', ' Dst', ' ap', ' AE'], loc='upper right', fontsize=15, framealpha=1)

plt.subplot(313)
color=['r', 'b', 'g', 'y', 'tan', 'c', 'm', 'pink']
for i in range(7):
    plt.plot(HLcor[i][10:30], 'o--', color=color[i], linewidth=2, markersize=12)
plt.xlim([3.97, 13.5])
plt.ylim([0, 1.03])
plt.yticks(fontsize=20)
plt.xlabel('offset   days', fontsize=25)
plt.ylabel('|LC|', fontsize=25)
plt.legend(['>2MeV', '>0.8MeV', '475keV', '275keV', '150keV', '75keV', '40keV'], fontsize=15, framealpha=1)
plt.xticks(np.arange(4,14,1),('0','1','2','3','4','5','6','7','8','9','10','11'), fontsize=20)
plt.show()
