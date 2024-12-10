import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# The path is hidden
data_path = r''
model_path = r''

timeStep = 20
timeStep_range= 30 
Distance = 100
step_train = 60
step_test = 60
outStep_gap = 1
time_range = timeStep * timeStep_range

df = pd.read_csv(data_path + '\\bus2_30.csv')
df['ECR'] = df['ECR']/3.6
df[['vel_std', 'acc_std', 'gas_std', 'bra_std', 'rot_std', 'tor_std']]=0

feature = ['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'ele', 'slope_real', 'vel', 'acc', 'gas_sta', 
            'bra_sta', 'v_rot', 'tor', 'vel_std', 'acc_std', 
            'gas_std', 'bra_std', 'rot_std', 'tor_std', 'ECR', 
            'dis']

seg_total = np.array(range((max(df['seg'])+1)))
seg_train, seg_test = train_test_split(seg_total, test_size=0.2, random_state = 20, shuffle = True)

xAll_train = list()
xAll_test = list()
Dis_true = list()
Step_true = list()
for i in range((max(df['seg'])+1)):
    df_temp = df[df['seg']==i][feature]
    df_temp = np.array(df_temp)
    if i in seg_train:
        step = step_train
    else:
        step = step_test
    for row in range(0, len(df_temp)-time_range, step):
        dis_true = 0
        outStep_true = 0
        outStep = 0
        while row+time_range+outStep<len(df_temp):
            dis_now = np.sum(df_temp[row+time_range:row+time_range+outStep,25])
            if dis_now>Distance:
                dis_true = dis_now
                outStep_true = outStep
                Step_true.append(outStep_true)
                Dis_true.append(dis_true)
                break
            outStep = outStep + outStep_gap
        if outStep_true == 0:
            break
        df_temp_xgb = df_temp[row+time_range-1,:25]
        df_temp_lstm = np.zeros((timeStep,len(feature)-1))
        for timestep in range(timeStep):
            for fea_dynamic in [12,13,14,15,16,17]:
                df_temp_lstm[timestep,fea_dynamic] = np.mean(df_temp[row+timestep \
                            *timeStep_range:row+(timestep+1)*timeStep_range,fea_dynamic])
            for fea_dynamic in [18,19,20,21,22,23]:
                df_temp_lstm[timestep,fea_dynamic] = np.std(df_temp[row+timestep \
                            *timeStep_range:row+(timestep+1)*timeStep_range,fea_dynamic-6])
        # ECR
        EC = np.sum(df_temp[row+time_range:row+time_range+outStep_true,24])
        df_temp_xgb[24] = EC/dis_true
        
        # static
        for fea in [7, 8]:
            df_temp_road = df_temp[row+time_range:row+time_range+outStep_true,fea]
            num_road = 0
            for j in range(len(df_temp_road)-1):    
                if (df_temp_road[j]==0) & (df_temp_road[j+1]==1):
                    num_road = num_road+1
            df_temp_xgb[fea] = num_road/dis_true*1000
        df_temp_xgb[9] = np.sum(df_temp[row+time_range:row+time_range+outStep_true,9]* \
                                  df_temp[row+time_range:row+time_range+outStep_true,25]/dis_true)
        df_temp_xgb[10] = np.sum(df_temp[row+time_range:row+time_range+outStep_true,11]* \
                                  df_temp[row+time_range:row+time_range+outStep_true,25]/dis_true)
        df_temp_xgb[11] = np.sqrt(np.sum((df_temp[row+time_range:row+time_range+outStep_true,25]/dis_true) \
                                   *(df_temp[row+time_range:row+time_range+outStep_true,11]-df_temp_xgb[10])**2))
        
        # dynamic
        for fea in [12,13,14,15,16,17]:
            df_temp_xgb[fea] = np.mean(df_temp[row+time_range:row+time_range+outStep_true,fea])
        for fea in [18,19,20,21,22,23]:
            df_temp_xgb[fea] = np.std(df_temp[row+time_range:row+time_range+outStep_true,fea-6])
        
        df_con = np.concatenate((df_temp_lstm, df_temp_xgb.reshape(1,-1)),axis=0)
        if i in seg_train:
            xAll_train.append(df_con)
        else:
            xAll_test.append(df_con)

xAll_train = np.array(xAll_train).reshape(-1,len(feature)-1)
xAll_test = np.array(xAll_test).reshape(-1,len(feature)-1)
split_train = -xAll_test.shape[0]
Data = np.concatenate((xAll_train, xAll_test), axis=0)

feature = ['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'slope', 'slope_std', 'vel', 'acc', 'gas_sta', 
            'bra_sta', 'v_rot', 'tor', 'vel_std', 'acc_std', 
            'gas_std', 'bra_std', 'rot_std', 'tor_std', 'ECR']

Data = pd.DataFrame(Data)
Data.columns = feature

inputCol = np.array(['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'slope', 'slope_std', 'vel', 'acc', 'gas_sta', 
            'bra_sta', 'v_rot', 'tor', 'vel_std', 'acc_std', 
            'gas_std', 'bra_std', 'rot_std', 'tor_std'])
inputCol_static = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
inputCol_dynamic = np.array([12,13,14,15,16,17, 18,19,20,21,22,23])

inputCol_select = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17, 18,19,20,21,22,23])

outputCol = np.array(['ECR'])

xAll = np.array(Data[inputCol])
yAll = np.array(Data[outputCol])

x_All_train = xAll[:split_train,:]
x_All_test = xAll[split_train:,:]
y_All_train = yAll[:split_train,:]
y_All_test = yAll[split_train:,:]

X_train = x_All_train.reshape(-1,timeStep+1,len(inputCol))
X_test = x_All_test.reshape(-1,timeStep+1,len(inputCol))
Y_train = y_All_train.reshape(-1,timeStep+1,len(outputCol))
Y_test = y_All_test.reshape(-1,timeStep+1,len(outputCol))