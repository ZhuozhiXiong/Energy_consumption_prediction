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
import xgboost as xgb


def LSTM_MLP(input_len_dynamic=1, timestep_dynamic=1, input_len_static=1, output_len=1):

    dynamic_input = Input(shape=(timestep_dynamic, input_len_dynamic))
    static_input1 = Input(shape=(input_len_static,))
    
    # RNN
    dynamic_lstm = LSTM(units=32, activation='relu')(dynamic_input)
    
    static_input = Concatenate(axis=1)([static_input1, dynamic_lstm])
    
    static_dense = Dense(units=32, activation='relu')(static_input)
    static_output = Dense(units=output_len, activation='linear', name='static_output')(static_dense)
    
    model = Model(inputs=[dynamic_input, static_input1], outputs=static_output)
    return model

def LSTM_Attention_MLP(input_len_dynamic=1, timestep_dynamic=1, input_len_static=1, output_len=1):

    dynamic_input = Input(shape=(timestep_dynamic, input_len_dynamic))
    static_input1 = Input(shape=(input_len_static,))
    
    # RNN
    dynamic_lstm = LSTM(units=32, activation='relu', return_sequences=True)(dynamic_input)
    
    # Attention
    dynamic_attention = tf.reduce_mean(dynamic_lstm,[2])
    dynamic_attention = Dense(units=4, activation='relu')(dynamic_attention)
    dynamic_attention = Dense(units=timestep_dynamic, activation='relu')(dynamic_attention)
    dynamic_attention= tf.nn.sigmoid(dynamic_attention)
    dynamic_attention = tf.reshape(dynamic_attention,[-1,timestep_dynamic,1])
    dynamic_attention = tf.multiply(dynamic_lstm, dynamic_attention)
    dynamic_attention = dynamic_lstm + dynamic_attention
    
    dynamic_conv = Flatten()(dynamic_attention)
    dynamic_output = Dense(units=32, activation='relu')(dynamic_conv)
    
    static_input = Concatenate(axis=1)([static_input1, dynamic_output])
    
    static_dense = Dense(units=32, activation='relu')(static_input)
    static_output = Dense(units=output_len, activation='linear', name='static_output')(static_dense)
    
    model = Model(inputs=[dynamic_input, static_input1], outputs=static_output)
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout, kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x, x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu", kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1, kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    return x + res

def model_dynamic(
    input_len_dynamic,
    timestep_dynamic,
    input_len_static,
    output_len,
    head_size=16,
    num_heads=2,
    num_transformer_blocks=2,
    ff_dim=16
):
    input_dynamic = Input(shape=(timestep_dynamic,input_len_dynamic))
    input_static = Input(shape=(input_len_static,))
    
    x = input_dynamic
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_dynamic = GlobalAveragePooling1D(data_format="channels_first")(x)
    
    inputs = Concatenate(axis=1)([input_static, output_dynamic])
    
    hid_dynamic = BatchNormalization()(inputs)
    hid_dynamic = Dense(units=64, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(hid_dynamic)
    hid_dynamic = Dropout(0.1)(hid_dynamic)
    outputs = Dense(units=output_len)(hid_dynamic)
    
    model = Model([input_dynamic, input_static], outputs)
    return model

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# The path is hidden
data_path = r''
model_path = r''

timeStep = 20
timeStep_range= 30
Distance = 100
step_train = 60
step_test = 300
outStep_gap = 1
time_range = timeStep * timeStep_range

df = pd.read_csv(data_path + '\\data_30.csv')
df['ECR'] = df['ECR']/3.6
df[['slope_max', 'slope_min']]=0

feature = ['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'ele', 'slope_real', 'slope_max', 'slope_min', 'vel', 
            'acc', 'gas_sta', 'bra_sta', 'v_rot', 'tor',
            'ECR', 'dis']

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
            dis_now = np.sum(df_temp[row+time_range:row+time_range+outStep,21])
            if (dis_now>Distance) & (dis_now<Distance*1.1):
                dis_true = dis_now
                outStep_true = outStep
                Step_true.append(outStep_true)
                Dis_true.append(dis_true)
                break
            outStep = outStep + outStep_gap
        if outStep_true == 0:
            break
        df_temp_xgb = df_temp[row+time_range-1,:21]
        df_temp_lstm = np.zeros((timeStep,len(feature)-1))
        for timestep in range(timeStep):
            for fea_dynamic in [14,15,16,17,18,19]:
                df_temp_lstm[timestep,fea_dynamic] = np.mean(df_temp[row+timestep \
                            *timeStep_range:row+(timestep+1)*timeStep_range,fea_dynamic])
        # ECR
        EC = np.sum(df_temp[row+time_range:row+time_range+outStep_true,20])
        df_temp_xgb[20] = EC/dis_true
        
        # static
        for fea in [7, 8]:
            df_temp_road = df_temp[row+time_range:row+time_range+outStep_true,fea]
            num_road = 0
            for j in range(len(df_temp_road)-1):    
                if (df_temp_road[j]==0) & (df_temp_road[j+1]==1):
                    num_road = num_road+1
            df_temp_xgb[fea] = num_road/dis_true
        df_temp_xgb[9] = np.mean(df_temp[row+time_range:row+time_range+outStep_true,9]* \
                                  df_temp[row+time_range:row+time_range+outStep_true,21])/dis_true
        ELE = np.sum(df_temp[row+time_range:row+time_range+outStep_true,10])
        df_temp_xgb[10] = ELE/dis_true
        df_temp_xgb[11] = np.std(df_temp[row+time_range:row+time_range+outStep_true,11]* \
                                  df_temp[row+time_range:row+time_range+outStep_true,21])/dis_true
        df_temp_xgb[12] = np.max(df_temp[row+time_range:row+time_range+outStep_true,11])
        df_temp_xgb[13] = np.min(df_temp[row+time_range:row+time_range+outStep_true,11])
        
        # dynamic
        for fea in [14,15,16,17,18,19]:
            df_temp_xgb[fea] = np.mean(df_temp[row+time_range:row+time_range+outStep_true,fea])
        
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
            'ECR']

x_All = pd.DataFrame(xAll)
x_All.columns = feature

inputCol = np.array(['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'ele', 'slope_std', 'slope_max', 'slope_min', 'vel', 
            'acc', 'gas_sta', 'bra_sta', 'v_rot', 'tor'])
inputCol_static = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
inputCol_dynamic = np.array([14, 15, 16, 17, 18, 19])
# inputCol_dynamic = np.array([14])

outputCol = np.array(['gas_sta'])

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

x_train_dynamic = X_train[:,timeStep-1,inputCol_dynamic]
x_train_static = np.concatenate((x_train_static, x_train_dynamic),axis=-1)
x_test_dynamic = X_test[:,timeStep-1,inputCol_dynamic]
x_test_static = np.concatenate((x_test_static, x_test_dynamic),axis=-1)

# Transformer-MLP

# positional embedding
# x_dynamic_position = np.arange(timeStep)/timeStep
# x_position = np.expand_dims(x_dynamic_position,0).repeat(len(inputCol_dynamic),axis=0).T
# x_train_position = np.expand_dims(x_position,0).repeat(x_train_dynamic.shape[0],axis=0)
# x_test_position = np.expand_dims(x_position,0).repeat(x_test_dynamic.shape[0],axis=0)
# x_train_dynamic = x_train_dynamic + x_train_position
# x_test_dynamic = x_test_dynamic + x_test_position

# model = model_dynamic(
#     input_len_dynamic=len(inputCol_dynamic),
#     timestep_dynamic=timeStep,
#     input_len_static=len(inputCol_static),
#     output_len=len(outputCol),
#     )

# model = LSTM_MLP(
#     input_len_dynamic=len(inputCol_dynamic),
#     timestep_dynamic=timeStep,
#     input_len_static=len(inputCol_static),
#     output_len=len(outputCol),
#     )

# model = LSTM_Attention_MLP(
#     input_len_dynamic=len(inputCol_dynamic),
#     timestep_dynamic=timeStep,
#     input_len_static=len(inputCol_static),
#     output_len=len(outputCol),
#     )

# model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
# # callbacks = [EarlyStopping(patience=30, restore_best_weights=True)]
# history = model.fit(
#     [x_train_dynamic, x_train_static], y_train,
#     batch_size=4096, epochs=500,
#     validation_data=([x_test_dynamic, x_test_static], y_test))
# history_train = history.history
# plt.figure()
# plt.plot(history_train['loss'][1:], label='Train')
# plt.plot(history_train['val_loss'][1:], label='test')
# plt.ylabel('MSE')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()
# plt.figure()
# plt.plot(history_train['loss'][100:], label='Train')
# plt.plot(history_train['val_loss'][100:], label='Test')
# plt.ylabel('MSE')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

model = xgb.XGBRegressor(max_depth=9, learning_rate=0.03, n_estimators=600, objective='reg:squarederror',
                        booster='gbtree', gamma=0, random_state=0)

model.fit(x_train_static, y_train, eval_set=[(x_train_static, y_train), (x_test_static, y_test)], eval_metric='rmse', 
          verbose=20)

# model.save(model_path + '\\transformer_' + str(Distance) + '_' + str(step_train) + '_' + str(timeStep) + '_' + str(timeStep_range) + '.h5')
# model = load_model(model_path + '\\transformer_' + str(Distance) + '_' + str(step_train) + '_' + str(timeStep) + '_' + str(timeStep_range) + '.h5')

Result = []

# test
# y_predict = model.predict([x_test_dynamic, x_test_static])
y_predict = model.predict(x_test_static)
y_p = Scaler_y.inverse_transform(y_predict.reshape(-1,1))
y_t = Scaler_y.inverse_transform(y_test.reshape(-1,1))

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

plt.figure()
plt.scatter(y_p, y_t, c='blue', s=5, alpha=0.8)
x_equal = [min(y_t), max(y_t)]
plt.plot(x_equal, x_equal, color='orange')
plt.show()

# train
# y_predict = model.predict([x_train_dynamic, x_train_static])
y_predict = model.predict(x_train_static)
y_p = Scaler_y.inverse_transform(y_predict.reshape(-1,1))
y_t = Scaler_y.inverse_transform(y_train.reshape(-1,1))

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