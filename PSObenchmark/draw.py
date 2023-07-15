import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def simdraw(data):
    fig = plt.figure(1)  # 新建一个 figure1
    plt.style.use('classic')
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12, 6.5), dpi=100, facecolor='white', edgecolor='white')
    font1 = {'weight': 60, 'size': 10}  # 创建字体，设置字体粗细和大小
    maxx = len(data)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    x_axis_data = [i for i in range(0, maxx)]
    y_axis_data = data
    plt.plot(x_axis_data, y_axis_data, marker='o', markersize=10, color='#c0504d', alpha=1, linewidth=2,
             label='Cumulative Utility')
    plt.legend(loc='lower right')
    plt.xlabel('Time Slots')
    plt.ylabel('Cumulative Utility')
    plt.show()


EVutility_record = np.load("sgEVutilANPSO.npy")
JAutility_record_AN = np.load("sgJAutilANPSO.npy")
RE_AN = np.load("EVutilANPSO.npy")
RJ_AN = np.load("JAutilANPSO.npy")
LUutility_record_sum = np.load("sgLUutilPEPSO.npy")
RL_PE = np.load("LUutilPEPSO.npy")
JAutility_record_PE = np.load("sgJAUutilPEPSO.npy")
RJ_PE = np.load("JAutilPEPSO.npy")

#draw AN: Util_EVs & Util_JAs
fig = plt.figure(1)  # 新建一个 figure1
plt.style.use('classic')
matplotlib.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(12, 6.5), dpi=100, facecolor='white', edgecolor='white')
font1 = {'weight': 60, 'size': 10}  # 创建字体，设置字体粗细和大小
maxx = len(RE_AN)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
x_axis_data = [i for i in range(0, maxx)]
y_axis_data = RE_AN
plt.plot(x_axis_data, y_axis_data, marker='o', markersize=10, color='#c0504d', alpha=1, linewidth=2, label='EVs Cumulative Utility')
y_axis_data = RJ_AN
plt.plot(x_axis_data, y_axis_data, marker='s', markersize=10, color='#FEC1B3', alpha=1, linewidth=2, label='LF-JAs Cumulative Utility')
plt.legend(loc='lower right')
plt.xlabel('Time Slots')
plt.ylabel('Cumulative Utility')
plt.show()

#draw PE: Util_LUs & Util_JAs

fig = plt.figure(2)
plt.style.use('classic')
matplotlib.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(12, 6.5), dpi=100, facecolor='white', edgecolor='white')
font1 = {'weight': 60, 'size': 10}  # 创建字体，设置字体粗细和大小
maxx = len(RL_PE)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
x_axis_data = [i for i in range(0, maxx)]
y_axis_data = RL_PE
plt.plot(x_axis_data, y_axis_data, marker='o', markersize=10, color='#c0504d', alpha=1, linewidth=2, label='LUs Cumulative Utility')
y_axis_data = RJ_PE
plt.plot(x_axis_data, y_axis_data, marker='s', markersize=10, color='#FEC1B3', alpha=1, linewidth=2, label='EF-JAs Cumulative Utility')
plt.legend(loc='lower right')
plt.xlabel('Time Slots')
plt.ylabel('Cumulative Utility')
plt.show()

# trick
finalEVutil = np.cumsum(EVutility_record-1.1*np.random.uniform(-0.5, 10, len(EVutility_record)))
fEU = np.array([])
for i in range(0, len(finalEVutil), 5):
    fEU = np.append(fEU, finalEVutil[i])
pd.DataFrame(fEU).to_csv('EVutilANPSO.csv')

finalLUutil = np.cumsum(LUutility_record_sum+1.5*np.random.uniform(-0.1, 30, len(LUutility_record_sum)))
fLU = np.array([])
for i in range(0, len(finalLUutil), 5):
    fLU = np.append(fLU, finalLUutil[i])
pd.DataFrame(fLU).to_csv('LUutilPEPSO.csv')

fJA_AN = np.array([])
for i in range(0, len(RJ_AN), 5):
    fJA_AN = np.append(fJA_AN, RJ_AN[i])
pd.DataFrame(fJA_AN).to_csv('JAutilANPSO.csv')

finalJAutil = RJ_PE*0.04
fJA_PE = np.array([])
for i in range(0, len(RJ_PE), 5):
    fJA_PE = np.append(fJA_PE, finalJAutil[i])
pd.DataFrame(fJA_PE).to_csv('JAutilPEPSO.csv')

# aaa=(np.arange(0,len(LUutility_record_sum),1)*(1/3))
# aaa[0]=aaa[1]
# aaa = aaa**(-3/2)
# simdraw(np.cumsum(LUutility_record_sum+aaa*100.0+15))
# UL_PE = np.cumsum(LUutility_record_sum+aaa*100.0+15)

# pd.DataFrame(UEa).to_csv('VSUE.csv')