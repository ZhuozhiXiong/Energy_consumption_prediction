import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import xgboost as xgb
import shap


def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train):
    
    Result = []
    
    # test
    y_p = y_p_test
    y_t = y_t_test
    
    SSR = np.sum((y_p-y_t)**2)
    SST = np.sum((y_t-np.mean(y_t))**2)
    r2 = 1-SSR/SST
    mae = np.sum(np.absolute(y_p-y_t))/len(y_t)
    mape = np.sum(np.absolute((y_p-y_t)/y_t))/len(y_t)
    mae_mean = mae/np.absolute(np.mean(y_t))
    rmse = (np.sum((y_p-y_t)**2)/len(y_t))**0.5
    Result.append(mae)
    Result.append(rmse)
    Result.append(r2)
    Result.append(mape)
    Result.append(mae_mean)

    # plt.figure()
    # plt.scatter(y_p, y_t, c='blue', s=5, alpha=0.8)
    # x_equal = [min(y_t), max(y_t)]
    # plt.plot(x_equal, x_equal, color='orange')
    # plt.show()

    # train
    y_p = y_p_train
    y_t = y_t_train

    SSR = np.sum((y_p-y_t)**2)
    SST = np.sum((y_t-np.mean(y_t))**2)
    r2 = 1-SSR/SST
    mae = np.sum(np.absolute(y_p-y_t))/len(y_t)
    mape = np.sum(np.absolute((y_p-y_t)/y_t))/len(y_t)
    mae_mean = mae/np.absolute(np.mean(y_t))
    rmse = (np.sum((y_p-y_t)**2)/len(y_t))**0.5
    Result.append(mae)
    Result.append(rmse)
    Result.append(r2)
    Result.append(mape)
    Result.append(mae_mean)

    Result = np.array(Result).reshape(1,-1)
    return Result

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# The path is hidden
data_path = r''
model_path = r''

timeStep = 1
timeStep_range= 300
Distance = 2000
step_train = 60
step_test = 60
outStep_gap = 10
time_range = timeStep * timeStep_range

df = pd.read_csv(data_path + '\\data_30.csv')
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
            if (dis_now>Distance) & (dis_now<Distance*1.1):
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
                            *timeStep_range:row+(timestep+1)*timeStep_range,fea_dynamic])
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
            df_temp_xgb[fea] = num_road/dis_true
        df_temp_xgb[9] = np.mean(df_temp[row+time_range:row+time_range+outStep_true,9]* \
                                  df_temp[row+time_range:row+time_range+outStep_true,25])/dis_true
        ELE = np.sum(df_temp[row+time_range:row+time_range+outStep_true,10])
        df_temp_xgb[10] = ELE/dis_true
        df_temp_xgb[11] = np.std(df_temp[row+time_range:row+time_range+outStep_true,11]* \
                                  df_temp[row+time_range:row+time_range+outStep_true,25])/dis_true
        
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
xAll = np.concatenate((xAll_train, xAll_test), axis=0)

feature = ['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'ele', 'slope_std', 'vel', 'acc', 'gas_sta', 
            'bra_sta', 'v_rot', 'tor', 'vel_std', 'acc_std', 
            'gas_std', 'bra_std', 'rot_std', 'tor_std', 'ECR']

x_All = pd.DataFrame(xAll)
x_All.columns = feature

inputCol = np.array(['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'ele', 'slope_std', 'vel', 'acc', 'gas_sta', 
            'bra_sta', 'v_rot', 'tor', 'vel_std', 'acc_std', 
            'gas_std', 'bra_std', 'rot_std', 'tor_std'])
inputCol_static = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
inputCol_dynamic = np.array([12,13,14,15,16,17, 18,19,20,21,22,23])
outputCol_dynamic = np.array([13])
inputCol_ECR = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13])

outputCol_ECR = np.array(['ECR'])

df = x_All

xAll = np.array(df[inputCol])
yAll = np.array(df[outputCol_ECR])

Scaler_x = MinMaxScaler(feature_range=(0,1))
Scaler_y = MinMaxScaler(feature_range=(0,1))
x_All = Scaler_x.fit_transform(xAll)
y_All = Scaler_y.fit_transform(yAll)

x_All_train = x_All[:split_train,:]
x_All_test = x_All[split_train:,:]
y_All_train = y_All[:split_train,:]
y_All_test = y_All[split_train:,:]

X_train = x_All_train.reshape(-1,timeStep+1,len(inputCol))
X_test = x_All_test.reshape(-1,timeStep+1,len(inputCol))
Y_train = y_All_train.reshape(-1,timeStep+1,len(outputCol_ECR))
Y_test = y_All_test.reshape(-1,timeStep+1,len(outputCol_ECR))

x_train_dynamic = np.concatenate((X_train[:,timeStep,inputCol_static], X_train[:,timeStep-1,inputCol_dynamic]), axis=-1)
y_train_dynamic = X_train[:,timeStep, outputCol_dynamic]
x_test_dynamic = np.concatenate((X_test[:,timeStep,inputCol_static], X_test[:,timeStep-1,inputCol_dynamic]), axis=-1)
y_test_dynamic = X_test[:,timeStep, outputCol_dynamic]

x_train_ECR = X_train[:,timeStep,inputCol_ECR]
y_train_ECR = Y_train[:,timeStep,:]
x_test_ECR = X_test[:,timeStep,inputCol_ECR]
y_test_ECR = Y_test[:,timeStep,:]

# dynamic prediction
model_dynamic = xgb.XGBRegressor(max_depth=9, learning_rate=0.04, n_estimators=500, objective='reg:squarederror',
                        booster='gbtree', gamma=0, random_state=0)

model_dynamic.fit(x_train_dynamic, y_train_dynamic, eval_set=[(x_train_dynamic, y_train_dynamic), (x_test_dynamic, y_test_dynamic)], eval_metric='rmse', 
          verbose=20)

# model_dynamic.save_model(model_path + '\\dynamic' + str(Distance) + '_' + str(step_train) + '.model')

y_p_test = model_dynamic.predict(x_test_dynamic).reshape(-1,1)
y_t_test = y_test_dynamic

y_p_train = model_dynamic.predict(x_train_dynamic).reshape(-1,1)
y_t_train = y_train_dynamic

Result_dynamic_predict = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)

# ECR regression
model_ECR = xgb.XGBRegressor(max_depth=5, learning_rate=0.04, n_estimators=400, objective='reg:squarederror',
                        booster='gbtree', gamma=0, random_state=0)

model_ECR.fit(x_train_ECR, y_train_ECR, eval_set=[(x_train_ECR, y_train_ECR), (x_test_ECR, y_test_ECR)], eval_metric='rmse', 
          verbose=20)

# model_ECR.save_model(model_path + '\\dynamic' + str(Distance) + '_' + str(step_train) + '.model')

importances = model_ECR.feature_importances_
feature_importance = pd.DataFrame({'feature':inputCol[inputCol_ECR], 'importance':importances})
feature_importance_sorted = feature_importance.sort_values(by='importance', ascending=True)

y_predict_test = model_ECR.predict(x_test_ECR)
y_p_test = Scaler_y.inverse_transform(y_predict_test.reshape(-1,1))
y_t_test = Scaler_y.inverse_transform(y_test_ECR.reshape(-1,1))

y_predict_train = model_ECR.predict(x_train_ECR)
y_p_train = Scaler_y.inverse_transform(y_predict_train.reshape(-1,1))
y_t_train = Scaler_y.inverse_transform(y_train_ECR.reshape(-1,1))

Result_ECR_regression = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)

# ECR prediction
dynamic_predict = model_dynamic.predict(x_test_dynamic)
x_test_ECR_predict = x_test_ECR.copy()
x_test_ECR_predict[:,[12]] = dynamic_predict.reshape(-1,1)
y_predict_test = model_ECR.predict(x_test_ECR_predict)
y_p_test = Scaler_y.inverse_transform(y_predict_test.reshape(-1,1))
y_t_test = Scaler_y.inverse_transform(y_test_ECR.reshape(-1,1))

dynamic_predict = model_dynamic.predict(x_train_dynamic)
x_train_ECR_predict = x_train_ECR.copy()
x_train_ECR_predict[:,[12]] = dynamic_predict.reshape(-1,1)
y_predict_train = model_ECR.predict(x_train_ECR_predict)
y_p_train = Scaler_y.inverse_transform(y_predict_train.reshape(-1,1))
y_t_train = Scaler_y.inverse_transform(y_train_ECR.reshape(-1,1))

Result_ECR_predict = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)

Result = np.concatenate((Result_dynamic_predict, Result_ECR_regression, Result_ECR_predict), axis=0)

# Feature importance
plt.figure()
plt.barh(np.arange(len(feature_importance_sorted)),feature_importance_sorted['importance'])
plt.yticks(np.arange(len(feature_importance_sorted)),feature_importance_sorted['feature'])
plt.grid()
plt.show()
