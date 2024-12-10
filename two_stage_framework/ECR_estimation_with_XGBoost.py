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
    
    # plt.figure()
    # plt.scatter(y_p, y_t, c='blue', s=5, alpha=0.8)
    # x_equal = [min(y_t), max(y_t)]
    # plt.plot(x_equal, x_equal, color='orange')
    # plt.show()
    
    SSR = np.sum((y_p-y_t)**2)
    SST = np.sum((y_t-np.mean(y_t))**2)
    r2 = 1-SSR/SST
    mae = np.sum(np.absolute(y_p-y_t))/len(y_t)
    mae_mean = mae/np.absolute(np.mean(y_t))
    rmse = (np.sum((y_p-y_t)**2)/len(y_t))**0.5
    
    y_p = y_p[np.absolute(y_t)>0.0001]
    y_t = y_t[np.absolute(y_t)>0.0001]
    mape = np.sum(np.absolute((y_p-y_t)/y_t))/len(y_t)
    Result.append(mae)
    Result.append(rmse)
    Result.append(r2)
    Result.append(mape)
    Result.append(mae_mean)
    
    # train
    y_p = y_p_train
    y_t = y_t_train

    SSR = np.sum((y_p-y_t)**2)
    SST = np.sum((y_t-np.mean(y_t))**2)
    r2 = 1-SSR/SST
    mae = np.sum(np.absolute(y_p-y_t))/len(y_t)
    mae_mean = mae/np.absolute(np.mean(y_t))
    rmse = (np.sum((y_p-y_t)**2)/len(y_t))**0.5
    
    y_p = y_p[np.absolute(y_t)>0.0001]
    y_t = y_t[np.absolute(y_t)>0.0001]
    mape = np.sum(np.absolute((y_p-y_t)/y_t))/len(y_t)
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

timeStep = 20
timeStep_range= 30
Distance = 16000
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

Scaler_x = MinMaxScaler(feature_range=(0,1))
Scaler_y = MinMaxScaler(feature_range=(0,1))
x_All_train = Scaler_x.fit_transform(x_All_train)
y_All_train = Scaler_y.fit_transform(y_All_train)
x_All_test = Scaler_x.transform(x_All_test)
y_All_test = Scaler_y.transform(y_All_test)

X_train = x_All_train.reshape(-1,timeStep+1,len(inputCol))
X_test = x_All_test.reshape(-1,timeStep+1,len(inputCol))
Y_train = y_All_train.reshape(-1,timeStep+1,len(outputCol))
Y_test = y_All_test.reshape(-1,timeStep+1,len(outputCol))

x_train = X_train[:,timeStep,inputCol_select]
y_train = Y_train[:,timeStep,:]
x_test = X_test[:,timeStep,inputCol_select]
y_test = Y_test[:,timeStep,:]

# XGBoost
model = xgb.XGBRegressor(max_depth=5, learning_rate=0.03, n_estimators=1000, objective='reg:squarederror',
                        booster='gbtree', gamma=0, random_state=0)

model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='rmse', 
          verbose=20)


model.save_model(model_path+'\\xgb_'+str(Distance)+'_'+str(timeStep)+'_'+str(timeStep_range)+'.model')

importances = model.feature_importances_
a_Feature_importances = importances.reshape(1,-1)

feature_importance = pd.DataFrame({'feature':inputCol[inputCol_select], 'importance':importances})
feature_importance_sorted = feature_importance.sort_values(by='importance', ascending=True)

thresholds = []
for importamce in importances:
    if importamce not in thresholds:
        thresholds.append(importamce)

thresholds = sorted(thresholds)

Result_select = []
Result_feature_select = []
Threshold_selcet = []
for threshold in thresholds:
    result_select = []
    selection = SelectFromModel(model, threshold=threshold, prefit=True)
    select_x_train = selection.transform(x_train)
    selection_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.03, n_estimators=1000, objective='reg:squarederror',
                            booster='gbtree', gamma=0, random_state=0)
    selection_model.fit(select_x_train,y_train,eval_metric='rmse', verbose=0)
    select_x_test = selection.transform(x_test)
    y_predict_test = selection_model.predict(select_x_test)
    y_p_test = Scaler_y.inverse_transform(y_predict_test.reshape(-1,1))
    y_t_test = Scaler_y.inverse_transform(y_test.reshape(-1,1))
    y_predict_train = selection_model.predict(select_x_train)
    y_p_train = Scaler_y.inverse_transform(y_predict_train.reshape(-1,1))
    y_t_train = Scaler_y.inverse_transform(y_train.reshape(-1,1))
    
    result_select = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)
    Result_select.append(result_select)
    Threshold_selcet.append([threshold,len(selection.get_support(True))])
    
    Result_feature_select.append(selection.get_support())
    # Result_feature_select.append(np.array(inputCol[selection.get_support(True)]).reshape(1,-1))
a_Result_select = np.array(Result_select).reshape(-1,10)
a_Result_feature_select = np.array(Result_feature_select)
a_Threshold_selcet = np.array(Threshold_selcet)

explainer = shap.TreeExplainer(model)
backgroud_select = np.arange(x_test.shape[0])
backgroud_select = np.random.choice(backgroud_select,size=100,replace=False)
backgroud = x_test[backgroud_select,:]
a_shap_values = explainer.shap_values(backgroud)
a_shap_values_feature = Scaler_x.inverse_transform(backgroud)
# importance = np.sum(np.absolute(shap_values), axis=0).T/shap_values.shape[0]
# importance_rate = importance/np.sum(importance)

# test
y_predict = model.predict(x_test)
y_p_test = Scaler_y.inverse_transform(y_predict.reshape(-1,1))
y_t_test = Scaler_y.inverse_transform(y_test.reshape(-1,1))
y_predict = model.predict(x_train)
y_p_train = Scaler_y.inverse_transform(y_predict.reshape(-1,1))
y_t_train = Scaler_y.inverse_transform(y_train.reshape(-1,1))
a_Result = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)

plt.figure()
plt.scatter(y_p_test, y_t_test, c='blue', s=5, alpha=0.8)
x_equal = [min(y_t_test), max(y_t_test)]
plt.plot(x_equal, x_equal, color='orange')
plt.show()

# Feature importance
plt.figure()
plt.barh(np.arange(len(feature_importance_sorted)),feature_importance_sorted['importance'])
plt.yticks(np.arange(len(feature_importance_sorted)),feature_importance_sorted['feature'])
plt.grid()
plt.show()
