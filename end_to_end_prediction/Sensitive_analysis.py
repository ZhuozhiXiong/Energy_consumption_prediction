import pandas as pd
import numpy as np
import time
import tensorflow as tf
from keras.models import  Model, load_model
from keras.layers import LSTM, GRU, Dense, Input, Reshape, Concatenate, Dropout, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention, BatchNormalization, Add, Activation, Conv1D, AveragePooling1D, Flatten
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError
from keras.initializers import TruncatedNormal
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
data_path = r''
model_path = r''

timeStep = 10
timeStep_range= 30
Distance = 500
step_train = 20
step_test = 20
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
            if (dis_now>Distance):
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
            node_index = []
            for j in range(len(df_temp_road)-1):    
                if (df_temp_road[j]==0) & (df_temp_road[j+1]==1):
                    if len(node_index) == 0:
                        node_index.append(j+1)
                        num_road = num_road+1
                    else:
                        if j+1 - node_index[-1]>120:
                            node_index.append(j+1)
                            num_road = num_road+1   
            df_temp_xgb[fea] = num_road
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

outputCol_ECR = np.array(['ECR'])

xAll = np.array(Data[inputCol])
yAll = np.array(Data[outputCol_ECR])

x_Static_train = xAll[:split_train,inputCol_static]
x_Dynamic_train = xAll[:split_train,inputCol_dynamic]
y_All_train = yAll[:split_train,:]

Scaler_x_static = MinMaxScaler(feature_range=(0,1))
Scaler_x_dynamic = MinMaxScaler(feature_range=(0,1))
Scaler_y = MinMaxScaler(feature_range=(0,1))
x_Static_train_scaler = Scaler_x_static.fit_transform(x_Static_train)
x_Dynamic_train = Scaler_x_dynamic.fit_transform(x_Dynamic_train)
y_All_train = Scaler_y.fit_transform(y_All_train)

x_Dynamic_train = x_Dynamic_train.reshape(-1,timeStep+1,len(inputCol_dynamic))
x_train_dynamic = x_Dynamic_train[:,:timeStep,:]

model = load_model(model_path + '\\transformer_' + str(Distance) + '_' + str(timeStep) + '_' + str(timeStep_range) + '.h5')

# sample_dynamic = x_train_dynamic[np.random.choice(x_train_dynamic.shape[0], size=20, replace=False), :]

# sample_dynamic_record = sample_dynamic.reshape(-1,len(inputCol_dynamic))
# sample_dynamic_record = Scaler_x_dynamic.inverse_transform(sample_dynamic_record)
# sample_dynamic_record = pd.DataFrame(sample_dynamic_record)
# sample_dynamic_record.to_csv(data_path + '\\sample_dynamic_record.csv', index=False)

sample_dynamic_record = pd.read_csv(data_path + '\\sample_dynamic_record.csv')
sample_dynamic_record = np.array(sample_dynamic_record)
sample_dynamic_record = Scaler_x_dynamic.transform(sample_dynamic_record)
sample_dynamic = sample_dynamic_record.reshape(-1,timeStep,len(inputCol_dynamic))

Input_feature_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # 
# x_static_mean = np.array([70, 20, 756, 55, 0, 0, 0, 0, 0, 1.8, 0, 2.5]).reshape(1,-1)
x_static_mean = np.array([99.2, 17, 756, 100, 1, 1, 0, 0, 0, 2, -0.979, 0.742]).reshape(1,-1)
x_static_change = {
    0: [50, 60, 70, 80, 90],
    1: [12, 16, 20, 24, 28],
    2: [748, 752, 756, 760, 764],
    3: [25, 40, 55, 70, 85],
    4: [0, 1],
    5: [0, 1],
    6: [0, 1],
    7: [0, 1, 2, 3],
    8: [0, 1, 2, 3],
    9: [1, 1.4, 1.8, 2.2, 2.6],
    10: [-2, -1, 0, 1, 2],
    11: [0.5, 1.5, 2.5, 3.5, 4.5],
      }

Analysis = dict()
for feature_select in Input_feature_select:
    Y_feature = np.zeros((20,len(x_static_change[feature_select])))
    for fea_change_index in range(len(x_static_change[feature_select])):
        x_static = x_static_mean.copy()
        x_static[0,feature_select] = x_static_change[feature_select][fea_change_index]
        x_static = Scaler_x_static.transform(x_static)
        for fea_dynamic_index in range(20):
            x_dynamic = sample_dynamic[fea_dynamic_index,:,:]
            y_predict = model.predict([x_dynamic.reshape(1,timeStep,len(inputCol_dynamic)), x_static])
            y_predict = Scaler_y.inverse_transform(y_predict.reshape(-1,1))
            Y_feature[fea_dynamic_index,fea_change_index] = y_predict
    Analysis[feature_select] = Y_feature
