import pandas as pd
import numpy as np
import time
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.keras.backend.get_session()
from keras.models import  Model, load_model
from keras.layers import LSTM, GRU, Dense, Input, Reshape, Concatenate, Dropout, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention, BatchNormalization, Add, Activation, Conv1D, AveragePooling1D, Flatten
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.svm import SVR
import shap


def Dynamic_Static(input_len_dynamic=1, timeStep=1, input_len_static=1, output_len=1):

    dynamic_input = Input(shape=(timeStep, input_len_dynamic))
    static_input1 = Input(shape=(input_len_static,))
    
    # RNN
    dynamic_lstm = LSTM(units=16, activation='relu', return_sequences=True)(dynamic_input)
    
    # Attention
    # dynamic_attention = tf.reduce_mean(dynamic_lstm,[2])
    # dynamic_attention = Dense(units=4, activation='relu')(dynamic_attention)
    # dynamic_attention = Dense(units=timeStep, activation='relu')(dynamic_attention)
    # dynamic_attention= tf.nn.sigmoid(dynamic_attention)
    # dynamic_attention = tf.reshape(dynamic_attention,[-1,timeStep,1])
    # dynamic_attention = tf.multiply(dynamic_lstm, dynamic_attention)
    # dynamic_attention = dynamic_lstm + dynamic_attention
    
    maxpool_channel = tf.reduce_max(dynamic_lstm,axis=2)
    avgpool_channel = tf.reduce_mean(dynamic_lstm,axis=2)
    mlp_1_max = Dense(timeStep/2, activation='relu')(maxpool_channel)
    mlp_2_max = Dense(timeStep, activation='relu')(mlp_1_max)
    mlp_2_max = tf.reshape(mlp_2_max, [-1,timeStep,1])
    mlp_1_avg = Dense(timeStep/2, activation='relu')(avgpool_channel)
    mlp_2_avg = Dense(timeStep, activation='relu')(mlp_1_avg)
    mlp_2_avg = tf.reshape(mlp_2_avg, [-1,timeStep,1])
    channel_attention= tf.nn.softmax(mlp_2_max + mlp_2_avg, axis=1)
    channel_attention = tf.multiply(dynamic_lstm, channel_attention)
    
    # maxpool_spatial = tf.reduce_max(dynamic_lstm,axis=1)
    # avgpool_spatial = tf.reduce_mean(dynamic_lstm,axis=1)
    # mlp_1_max = Dense(16/2, activation='relu')(maxpool_spatial)
    # mlp_2_max = Dense(16, activation='relu')(mlp_1_max)
    # mlp_2_max = tf.reshape(mlp_2_max, [-1,1,16])
    # mlp_1_avg = Dense(16/2, activation='relu')(avgpool_spatial)
    # mlp_2_avg = Dense(16, activation='relu')(mlp_1_avg)
    # mlp_2_avg = tf.reshape(mlp_2_avg, [-1,1,16])
    # spatial_attention= tf.nn.softmax(mlp_2_max + mlp_2_avg, axis=2)
    # dynamic_attention = tf.multiply(dynamic_lstm, spatial_attention)
    
    dynamic_attention = dynamic_lstm + channel_attention
    
    # Resnet
    # dynamic_conv = Conv1D(4, 1, activation='relu', padding='same')(dynamic_attention)
    # dynamic_conv = BatchNormalization()(dynamic_conv)
    # dynamic_conv = Conv1D(4, 3, activation='relu', padding='same')(dynamic_conv)
    # dynamic_conv = BatchNormalization()(dynamic_conv)
    # dynamic_conv = Conv1D(16, 1, padding='same')(dynamic_conv)
    # dynamic_conv = Add()([dynamic_attention, dynamic_conv])
    # dynamic_conv = tf.nn.relu(dynamic_conv)
    
    dynamic_conv = Flatten()(dynamic_attention)
    dynamic_output = Dense(units=12, activation='relu')(dynamic_conv)
    
    static_input = Concatenate(axis=1)([static_input1, dynamic_output])
    
    # Attention
    # static_attention = Dense(units=4, activation='relu')(static_input)
    # static_attention = Dense(units=12+14, activation='relu')(static_attention)
    # static_attention= tf.nn.sigmoid(static_attention)
    # static_attention = tf.multiply(static_input, static_attention)
    # static_attention = static_input + static_attention
    
    static_dense = Dense(units=16, activation='relu')(static_input)
    # static_dense = Dense(units=4, activation='relu')(static_dense)
    static_output = Dense(units=output_len, activation='linear', name='static_output')(static_dense)
    
    model = Model(inputs=[dynamic_input, static_input1], outputs=static_output)
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def model_dynamic(
    input_len_dynamic,
    timestep_dynamic,
    input_len_static,
    output_len,
    head_size=1,
    num_heads=3,
    num_transformer_blocks=2,
    ff_dim=16
):
    input_dynamic = Input(shape=(timestep_dynamic,input_len_dynamic))
    input_static = Input(shape=(input_len_static,))
    
    x = input_dynamic
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.1)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    # output_dynamic = Dense(32, activation="relu")(x)
    # output_dynamic = Dense(16, activation="relu")(x)
    output_dynamic = x
    
    inputs = Concatenate(axis=1)([input_static, output_dynamic])
    
    hid_dynamic = BatchNormalization()(inputs)
    hid_dynamic = Dense(units=32, activation='relu')(hid_dynamic)
    hid_dynamic = BatchNormalization()(hid_dynamic)
    hid_dynamic = Dense(units=8, activation='relu')(hid_dynamic)
    hid_dynamic = BatchNormalization()(hid_dynamic)
    outputs = Dense(units=output_len, activation='relu')(hid_dynamic)
    
    model = Model([input_dynamic, input_static], outputs)
    return model

def LSTM_MLP(input_len_dynamic=1, timeStep=1, input_len_static=1, output_len=1):

    dynamic_input = Input(shape=(timeStep, input_len_dynamic))
    static_input1 = Input(shape=(input_len_static,))
    
    # RNN
    dynamic_lstm = LSTM(units=16, activation='relu')(dynamic_input)
    dynamic_output = Dense(units=12, activation='relu')(dynamic_lstm)
    
    static_input = Concatenate(axis=1)([static_input1, dynamic_output])
    
    static_dense = Dense(units=16, activation='relu')(static_input)
    static_dense = Dense(units=4, activation='relu')(static_dense)
    static_output = Dense(units=output_len, activation='linear', name='static_output')(static_dense)
    
    model = Model(inputs=[dynamic_input, static_input1], outputs=static_output)
    return model

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
data_path = r''
model_path = r''

timeStep = 20
timeStep_range= 20
Distance = 500
step_train = 150
step_test = 300
outStep_gap = 1
time_range = timeStep * timeStep_range

df = pd.read_csv(data_path + '\\data_30.csv')
df['ECR'] = df['ECR']/3.6
df[['slope_max', 'slope_min','vel_std', 'acc_std', 'gas_std', 'bra_std', 'rot_std', 'tor_std']]=0

feature = ['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'ele', 'slope_real', 'slope_max', 'slope_min', 'vel', 
            'acc', 'gas_sta', 'bra_sta', 'v_rot', 'tor',
            'vel_std', 'acc_std', 'gas_std', 'bra_std', 'rot_std', 
            'tor_std', 'ECR', 'dis']

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
            dis_now = np.sum(df_temp[row+time_range:row+time_range+outStep,27])
            if (dis_now>Distance): #  & (dis_now<Distance*1.1)
                dis_true = dis_now
                outStep_true = outStep
                Step_true.append(outStep_true)
                Dis_true.append(dis_true)
                break
            outStep = outStep + outStep_gap
        if outStep_true == 0:
            break
        df_temp_xgb = df_temp[row+time_range-1,:27]
        df_temp_lstm = np.zeros((timeStep,len(feature)-1))
        for timestep in range(timeStep):
            for fea_dynamic in [14,15,16,17,18,19]:
                df_temp_lstm[timestep,fea_dynamic] = np.mean(df_temp[row+timestep \
                            *timeStep_range:row+(timestep+1)*timeStep_range,fea_dynamic])
            for fea_dynamic in [20,21,22,23,24,25]:
                df_temp_lstm[timestep,fea_dynamic] = np.std(df_temp[row+timestep \
                            *timeStep_range:row+(timestep+1)*timeStep_range,fea_dynamic-6])
        # ECR
        EC = np.sum(df_temp[row+time_range:row+time_range+outStep_true,26])
        df_temp_xgb[26] = EC/dis_true
        
        # static
        for fea in [7, 8]:
            df_temp_road = df_temp[row+time_range:row+time_range+outStep_true,fea]
            num_road = 0
            for j in range(len(df_temp_road)-1):    
                if (df_temp_road[j]==0) & (df_temp_road[j+1]==1):
                    num_road = num_road+1
            df_temp_xgb[fea] = num_road/dis_true
        df_temp_xgb[9] = np.mean(df_temp[row+time_range:row+time_range+outStep_true,9]* \
                                  df_temp[row+time_range:row+time_range+outStep_true,27])/dis_true
        ELE = np.sum(df_temp[row+time_range:row+time_range+outStep_true,10])
        df_temp_xgb[10] = ELE/dis_true
        df_temp_xgb[11] = np.std(df_temp[row+time_range:row+time_range+outStep_true,11]* \
                                  df_temp[row+time_range:row+time_range+outStep_true,27])/dis_true
        df_temp_xgb[12] = np.max(df_temp[row+time_range:row+time_range+outStep_true,11])
        df_temp_xgb[13] = np.min(df_temp[row+time_range:row+time_range+outStep_true,11])
        
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
            'ele', 'slope_std', 'slope_max', 'slope_min', 'vel', 
            'acc', 'gas_sta', 'bra_sta', 'v_rot', 'tor',
            'vel_std', 'acc_std', 'gas_std', 'bra_std', 'rot_std', 
            'tor_std', 'ECR']

x_All = pd.DataFrame(xAll)
x_All.columns = feature

# x_All.to_csv(data_path + '\\dis1_step120.csv',index=False)

inputCol = np.array(['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'ele', 'slope_std', 'slope_max', 'slope_min', 'vel', 
            'acc', 'gas_sta', 'bra_sta', 'v_rot', 'tor',
            'vel_std', 'acc_std', 'gas_std', 'bra_std', 'rot_std', 
            'tor_std'])
inputCol_static = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
# inputCol_dynamic = np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
inputCol_dynamic = np.array([14, 15])

outputCol = np.array(['ECR'])

# df = pd.read_csv(data_path + '\\dis1_step150.csv')
# split_train = -39291
df = x_All

xAll = np.array(df[inputCol])
yAll = np.array(df[outputCol])

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
Y_train = y_All_train.reshape(-1,timeStep+1,len(outputCol))
Y_test = y_All_test.reshape(-1,timeStep+1,len(outputCol))

x_train_dynamic = X_train[:,:timeStep,inputCol_dynamic]
x_train_static = X_train[:,timeStep,inputCol_static]
y_train = Y_train[:,timeStep,:]
x_test_dynamic = X_test[:,:timeStep,inputCol_dynamic]
x_test_static = X_test[:,timeStep,inputCol_static]
y_test = Y_test[:,timeStep,:]

# XGBoost
# model = xgb.XGBRegressor(max_depth=7,
#                         learning_rate=0.05,
#                         n_estimators=200,
#                         objective='reg:squarederror',
#                         booster='gbtree',
#                         gamma=0,
#                         min_child_weight=1,
#                         subsample=0.9,
#                         colsample_bytree=0.9,
#                         reg_alpha=0,
#                         reg_lambda=1,
#                         random_state=0)

# x_train = np.concatenate((x_train_static,np.mean(x_train_dynamic,axis=1)),axis=1)
# x_test = np.concatenate((x_test_static,np.mean(x_test_dynamic,axis=1)),axis=1)
# model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='rmse', 
#           verbose=5)
# y_predict = model.predict(x_test)

# model.fit(x_train_static, y_train, eval_set=[(x_train_static, y_train), (x_test_static, y_test)], eval_metric='rmse', 
#           verbose=5)
# y_predict = model.predict(x_test_static)

# RF
# model = RandomForestRegressor(n_estimators=100, random_state=1)
# model.fit(x_train_static, y_train.reshape(-1))
# y_predict = model.predict(x_test_static)

# SVM
# model = GridSearchCV(SVR(kernel='rbf'),param_grid={'C': [0.2, 1, 5],'gamma': [0.05, 0.1, 0.5]}, cv=3)
# # model = SVR(kernel='rbf')
# model.fit(x_train_static, y_train.reshape(-1))
# y_predict = model.predict(x_test_static)

# MLP
# input_BPNN = Input(shape=(x_train_static.shape[-1]),)
# x = Dense(128,activation='relu')(input_BPNN)
# x = BatchNormalization()(x)
# x = Dense(256,activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dense(128,activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dense(64,activation='relu')(x)
# x = BatchNormalization()(x)
# output_BPNN = Dense(y_train.shape[-1],activation='linear')(x)
# model = Model(input_BPNN,output_BPNN)
# model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# callbacks = [EarlyStopping(patience=200, restore_best_weights=True)]
# history = model.fit(x_train_static, y_train, batch_size=256, epochs=1000,
#     validation_data=(x_test_static, y_test), callbacks=callbacks)
# history_train = history.history
# plt.figure()
# plt.plot(history_train['loss'][10:], label='训练集')
# plt.plot(history_train['val_loss'][10:], label='验证集')
# plt.ylabel('MSE')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()
# y_predict = model.predict(x_test_static)

# LSTM
# input_lstm = Input(shape=(timeStep,x_train_dynamic.shape[-1]))
# x = LSTM(16,activation='relu')(input_lstm)
# x = Dense(16,activation='relu')(x)
# x = Dense(4,activation='relu')(x)
# output_lstm = Dense(y_train.shape[-1],activation='linear')(x)
# model = Model(input_lstm,output_lstm)
# model.compile(loss="mse", optimizer=Adam(), metrics=[RootMeanSquaredError()])
# callbacks = [EarlyStopping(patience=20, restore_best_weights=True)]
# history = model.fit(x_train_dynamic, y_train, batch_size=2000, epochs=500,
#     validation_data=(x_test_dynamic, y_test), callbacks=callbacks)
# history_train = history.history
# plt.figure()
# plt.plot(history_train['loss'][50:], label='train')
# plt.plot(history_train['val_loss'][50:], label='test')
# plt.ylabel('MSE')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()
# y_predict = model.predict(x_test_dynamic)

# Transformer-MLP
model = model_dynamic(
    input_len_dynamic=len(inputCol_dynamic),
    timestep_dynamic=timeStep,
    input_len_static=len(inputCol_static),
    output_len=len(outputCol),
    )

# LSTM-MLP
# model = LSTM_MLP(
#     input_len_dynamic=len(inputCol_dynamic),
#     timeStep=timeStep, 
#     input_len_static=len(inputCol_static),
#     output_len=len(outputCol),
#     )

# Attention
# model = Dynamic_Static(
#     input_len_dynamic=len(inputCol_dynamic),
#     timeStep=timeStep, 
#     input_len_static=len(inputCol_static),
#     output_len=len(outputCol),
#     )

model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
callbacks = [EarlyStopping(patience=50, restore_best_weights=True)]
history = model.fit(
    [x_train_dynamic, x_train_static], y_train,
    batch_size=2048, epochs=1500,
    validation_data=([x_test_dynamic, x_test_static], y_test), callbacks=callbacks)
history_train = history.history
plt.figure()
plt.plot(history_train['loss'][20:], label='train')
plt.plot(history_train['val_loss'][20:], label='test')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()
y_predict = model.predict([x_test_dynamic, x_test_static])

background1 = x_train_dynamic[np.random.choice(x_train_dynamic.shape[0], 100, replace=False)]
background2 = x_train_static[np.random.choice(x_train_static.shape[0], 100, replace=False)]
explainer1 = shap.DeepExplainer(model, [background1, background2])
shap_values1 = explainer1.shap_values([x_test_dynamic[:100,:,:], x_test_static[:100,:]])
shap_values1_2D = shap_values1[0].reshape(-1,len(inputCol_dynamic)+len(inputCol_static))
x_test1_2D = x_test_dynamic[:100,:,:].reshape(-1,len(inputCol_dynamic))
x_test1_2d = pd.DataFrame(data=x_test1_2D, columns = inputCol[inputCol_dynamic])
x_test2_2D = x_test_static[:100,:].reshape(-1,len(inputCol_static))
x_test2_2d = pd.DataFrame(data=x_test2_2D, columns = inputCol[inputCol_static])
plt.figure()
shap.summary_plot(shap_values1_2D, [x_test1_2d, x_test2_2d])
plt.figure()
shap.summary_plot(shap_values1_2D, [x_test1_2d, x_test2_2d], plot_type="bar")

# importance1 = np.sum(np.absolute(shap_values1_2D), axis=0).T/shap_values1_2D.shape[0]

# model.save(model_path + '\\transformer_60_1_5_6.h5')
# model = load_model(model_path + '\\transformer_30_1_5_6.h5')

y_p = Scaler_y.inverse_transform(y_predict.reshape(-1,1))
y_t = Scaler_y.inverse_transform(y_test.reshape(-1,1))

MAE = list()
RMSE = list()
R2 = list()
MAPE = list()
MAE_MEAN = list()
Result = list()

SSR = np.sum((y_p-y_t)**2)
SST = np.sum((y_t-np.mean(y_t))**2)
r2 = 1-SSR/SST
mae = np.sum(np.absolute(y_p-y_t))/len(y_t)
mape = np.sum(np.absolute((y_p-y_t)/y_t))/len(y_t)
mae_mean = mae/np.absolute(np.mean(y_t))
rmse = (np.sum((y_p-y_t)**2)/len(y_t))**0.5
MAE.append(mae)
RMSE.append(rmse)
R2.append(r2)
MAPE.append(mape)
MAE_MEAN.append(mae_mean)

Result.append(MAE)
Result.append(RMSE)
Result.append(R2)
Result.append(MAPE)
Result.append(MAE_MEAN)
Result = np.array(Result).T

plt.figure()
plt.scatter(y_p, y_t, c='blue', s=5, alpha=0.8)
x_equal = [min(y_t), max(y_t)]
plt.plot(x_equal, x_equal, color='orange')
plt.show()