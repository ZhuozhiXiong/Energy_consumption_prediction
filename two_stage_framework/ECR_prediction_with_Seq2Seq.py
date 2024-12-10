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
from keras.layers import Activation,  Lambda, RepeatVector, Dot
from keras import backend as K


def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
        
def one_step_attention(a, s_prev, repeator, concatenator, densor, activator, dotor):
    
    s_prev = repeator(s_prev)
    
    concat = concatenator([s_prev, a])
    e = densor(concat)
    alphas = activator(e)
    
    context =  dotor([alphas, a])
    return context

def seq2seq_attention(input_len=1, timeStep=1, output_len=1, outStep=1, unit_lstm=10):

    X = Input(shape=(timeStep,input_len))

    all_outputs = []

    # Encoder: pre-attention LSTM
    encoder = LSTM(units=unit_lstm, return_state=True, return_sequences=True, name='encoder')
    # Decoder: post-attention LSTM
    decoder = LSTM(units=unit_lstm, return_state=True, name='decoder')
    # Output
    decoder_hid1 = Dense(units=64, activation='relu', name='decoder_hid1')
    decoder_hid2 = Dense(units=32, activation='relu', name='decoder_hid2')
    decoder_hid3 = Dense(units=16, activation='relu', name='decoder_hid3')
    decoder_output = Dense(units=output_len, activation='softmax', name='output')
    model_output = Reshape((1, output_len))

    # Attention
    repeator = RepeatVector(timeStep)
    concatenator = Concatenate(axis=-1)
    densor = Dense(1, activation = "relu")
    activator = Activation(softmax, name='attention_weights')
    dotor =  Dot(axes = 1)

    encoder_outputs, s, c = encoder(X)

    for t in range(outStep):
        context = one_step_attention(encoder_outputs, s, repeator, concatenator, densor, activator, dotor)

        a, s, c = decoder(context, initial_state=[s, c])

        outputs = decoder_output(decoder_hid3(decoder_hid2(decoder_hid1(a))))
        outputs = model_output(outputs)
        all_outputs.append(outputs)

    all_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model(inputs=X, outputs=all_outputs)

    return model


def model_seq2seq(input_dynamic_len=1, input_static_len=1, timeStep=1, output_len=1, outStep=1):

    X_dynamic = Input(shape=(timeStep,input_dynamic_len))
    X_static = Input(shape=(input_static_len,))

    all_outputs = []

    # Encoder: pre-attention LSTM
    encoder = LSTM(units=10, return_state=True, return_sequences=True, name='encoder')
    # Decoder: post-attention LSTM
    decoder = LSTM(units=10, return_state=True, name='decoder')
    # Output
    decoder_hid1 = Dense(units=64, activation='relu', name='decoder_hid1')
    decoder_hid2 = Dense(units=32, activation='relu', name='decoder_hid2')
    decoder_hid3 = Dense(units=16, activation='relu', name='decoder_hid3')
    decoder_output = Dense(units=output_len, activation='softmax', name='output')
    model_output = Reshape((1, output_len))

    # Attention
    repeator = RepeatVector(timeStep)
    concatenator = Concatenate(axis=-1)
    densor = Dense(1, activation = "relu")
    activator = Activation(softmax, name='attention_weights')
    dotor =  Dot(axes = 1)

    encoder_outputs, s, c = encoder(X_dynamic)

    for t in range(outStep):
        context = one_step_attention(encoder_outputs, s, repeator, concatenator, densor, activator, dotor)

        a, s, c = decoder(context, initial_state=[s, c])
        
        X_con = Concatenate(axis=1)([a, X_static])

        outputs = decoder_output(decoder_hid3(decoder_hid2(decoder_hid1(X_con))))
        outputs = model_output(outputs)
        all_outputs.append(outputs)

    all_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model(inputs=[X_dynamic, X_static], outputs=all_outputs)

    return model

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
Distance = 250
step_train = 600
step_test = 300
outStep_gap = 1
time_range = timeStep * timeStep_range

batch_size = 4096
epochs = 100

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
Y_train = y_All_train.reshape(-1,timeStep+1,len(outputCol_ECR))
Y_test = y_All_test.reshape(-1,timeStep+1,len(outputCol_ECR))

x_train_static = X_train[:,timeStep,inputCol_static]
x_train_dynamic = X_train[:,:timeStep,inputCol_dynamic]
x_test_static = X_test[:,timeStep,inputCol_static]
x_test_dynamic = X_test[:,:timeStep,inputCol_dynamic]

y_train_ECR = Y_train[:,timeStep,:]
y_test_ECR = Y_test[:,timeStep,:]

# Transformer-MLP

# positional embedding
x_dynamic_position = np.arange(timeStep)/timeStep
x_position = np.expand_dims(x_dynamic_position,0).repeat(len(inputCol_dynamic),axis=0).T
x_train_position = np.expand_dims(x_position,0).repeat(x_train_dynamic.shape[0],axis=0)
x_test_position = np.expand_dims(x_position,0).repeat(x_test_dynamic.shape[0],axis=0)
x_train_dynamic = x_train_dynamic + x_train_position
x_test_dynamic = x_test_dynamic + x_test_position

model = seq2seq_attention(
    input_len=len(inputCol_dynamic), 
    timeStep=timeStep, 
    output_len=1, 
    outStep=1, 
    unit_lstm=10)

model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
# callbacks = [EarlyStopping(patience=30, restore_best_weights=True)]
history = model.fit(
    x_train_dynamic, y_train_ECR,
    batch_size=batch_size, epochs=epochs,
    validation_data=(x_test_dynamic, y_test_ECR))
history_train = history.history
plt.figure()
plt.plot(history_train['loss'][1:], label='Train')
plt.plot(history_train['val_loss'][1:], label='Test')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()
plt.figure()
plt.plot(history_train['loss'][100:], label='Train')
plt.plot(history_train['val_loss'][100:], label='Test')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.save(model_path + '\\transformer_' + str(Distance) + '_' + str(timeStep) + '_' + str(timeStep_range) + '.h5')
# model = load_model(model_path + '\\transformer_' + str(Distance) + '_' + str(timeStep) + '_' + str(timeStep_range) + '.h5')

y_predict_test = model.predict([x_test_dynamic, x_test_static])
y_p_test = Scaler_y.inverse_transform(y_predict_test.reshape(-1,1))
y_t_test = Scaler_y.inverse_transform(y_test_ECR.reshape(-1,1))
y_predict_train = model.predict([x_train_dynamic, x_train_static])
y_p_train = Scaler_y.inverse_transform(y_predict_train.reshape(-1,1))
y_t_train = Scaler_y.inverse_transform(y_train_ECR.reshape(-1,1))

a_Result = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)
a_data_predict = np.concatenate((y_p_test, y_t_test), axis=1)