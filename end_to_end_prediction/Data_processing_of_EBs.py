import pandas as pd
import numpy as np
from scipy import interpolate
from geographiclib.geodesic import Geodesic
import time

path_data = r''
# df1 = pd.read_csv(path_data + '\\09_ful.csv')
# df2 = pd.read_csv(path_data + '\\16_ful.csv')
# df3 = pd.read_csv(path_data + '\\19_ful.csv')
# df4 = pd.read_csv(path_data + '\\21_ful.csv')
# df5 = pd.read_csv(path_data + '\\22_ful.csv')
# df6 = pd.read_csv(path_data + '\\26_ful.csv')
# df7 = pd.read_csv(path_data + '\\28_ful.csv')
# df8 = pd.read_csv(path_data + '\\38_ful.csv')
# df9 = pd.read_csv(path_data + '\\54_ful.csv')

# df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], ignore_index=True)

# df['formate_time'] = pd.to_datetime(df['时间'],format="%Y/%m/%d %H:%M:%S.010.")
# df['index'] = (df['formate_time'].astype('int64')//1e9).astype("int")

# Feature_all = ['formate_time', 'index', 'GPS经度','GPS纬度','GPS车速','GPS方向','整车状态','充电状态','动力电池总电压','动力电池总电流',
#                 'SOC','油门踏板状态','制动踏板状态','驱动电机转速','驱动电机扭矩']

# df_feature = df[Feature_all]

# df_feature.to_csv(path_data + '\\data_raw.csv',index=False)

# df1 = pd.read_csv(path_data + '\\data_raw.csv')
# df1['功率'] = df1['动力电池总电流'] * df1['动力电池总电压'] / 1000

# df1 = df1.dropna(how='any') # 删除缺失数据
# df1 = df1.drop_duplicates() # 删除重复数据
# df1 = df1.drop(df1[(df1['GPS经度']<1) | (df1['GPS纬度']<1)].index) # 删除异常GPS数据
# df1 = df1.drop(df1[df1['整车状态']<0.5].index) # 删除车辆未发动数据
# df1 = df1.drop(df1[df1['充电状态']>0.5].index) # 删除车辆充电数据
# df1 = df1.drop(df1[(df1['GPS车速']>70) | (df1['GPS车速']<0)].index)
# df1 = df1.drop(df1[df1['驱动电机扭矩']>3000].index)
# df1 = df1.drop(df1[df1['功率']>200].index)

# df1.to_csv(path_data + '\\data_remove.csv',index=False)

df1 = pd.read_csv(path_data + '\\road_fus.csv')

Feature_all = ['index', 'formate_time', 'GPS经度','GPS纬度','GPS车速','GPS方向','整车状态','充电状态','动力电池总电压','动力电池总电流',
                'SOC','油门踏板状态','制动踏板状态','驱动电机转速','驱动电机扭矩','road_level','bus_stop','inter','slope_real','hight']
df1 = df1[Feature_all]

dic_fea = dict()
for i in range(len(Feature_all)):
    dic_fea[Feature_all[i]] = i

# 删除停车时间过长的数据
time_start = time.time()
print('正在删除停车时间过长数据')

threshold = 300
df1 = df1.reset_index(drop=True)
length1 = len(df1)
start_stop = 0
flag_start = 0
if df1['GPS车速'][0]<1:
    start_stop = 0
    flag_start = 1
end_stop = 1
flag_end = 0
for i in np.arange(1,length1):
    if (df1['GPS车速'][i]<1) & (df1['GPS车速'][i-1]>=1):
        start_stop = i
        flag_start = 1
    if (df1['GPS车速'][i]>=1) & (df1['GPS车速'][i-1]<1):
        end_stop = i
        flag_end = 1
    if flag_start*flag_end >0.5:
        time_stop = df1['index'][end_stop] - df1['index'][start_stop]
        flag_start = 0
        flag_end = 0
        if time_stop>threshold:
            df1 = df1.drop(np.arange(start_stop,end_stop))

# threshold = 300
# array1 = np.array(df1)
# index_del = np.array([])
# length1 = array1.shape[0]
# start_stop = 0
# flag_start = 0
# if array1[0,dic_fea['GPS车速']]<1:
#     start_stop = 0
#     flag_start = 1
# end_stop = 1
# flag_end = 0
# for i in np.arange(1,length1):
#     if (array1[i,dic_fea['GPS车速']]<1) & (array1[i-1,dic_fea['GPS车速']]>=1):
#         start_stop = i
#         flag_start = 1
#     if (array1[i,dic_fea['GPS车速']]>=1) & (array1[i-1,dic_fea['GPS车速']]<1):
#         end_stop = i
#         flag_end = 1
#     if flag_start*flag_end >0.5:
#         time_stop = array1[end_stop,dic_fea['index']] - array1[start_stop,dic_fea['index']]
#         flag_start = 0
#         flag_end = 0
#         if time_stop>threshold:
#             index_del = np.append(index_del, np.arange(start_stop,end_stop))
# index_del = np.array(index_del,dtype='int')
# df1 = df1.drop(index_del)

print('删除停车时间过长数据完成')
time_end = time.time()
print('删除停车时间过长数据用时%d秒' %(time_end-time_start))
print('数据总样本为%d条' %(len(df1)))

# 划分片段
time_start = time.time()
print('正在划分片段')

df1 = df1.reset_index(drop=True)
df1['segment'] = 0
seg = 0
length1 = len(df1)
threshold_seg = 30 # 片段划分阈值
for i in np.arange(1,length1):
    if df1['index'][i]-df1['index'][i-1] > threshold_seg:
        seg = seg + 1
        df1.loc[i, 'segment'] = seg
    else:
        df1.loc[i, 'segment'] = seg

# seg = 0
# length1 = array2.shape[0]
# threshold_seg = 10 # 片段划分阈值
# for i in np.arange(1,length1):
#     if array2[i,dic_fea['index']]-array2[i-1,dic_fea['index']] > threshold_seg:
#         seg = seg + 1
#         array2[i,dic_fea['segment']] = seg
#     else:
#         array2[i,dic_fea['segment']] = seg

print('划分片段完成')
time_end = time.time()
print('划分片段用时%d秒' %(time_end-time_start))

# 删除时间异常片段
time_start = time.time()
print('正在删除时间异常片段')

Num_seg = max(df1['segment'])
threshold_time = 120
for i in range(Num_seg+1):
    df_temp = df1[df1['segment']==i]
    range_time = df_temp.iloc[-1]['index'] - df_temp.iloc[0]['index']
    range_len = len(df_temp)
    if (range_time<threshold_time) | (range_len<0.8*range_time):
        df1 = df1.drop(df1[df1['segment']==i].index)

# df1 = pd.DataFrame(array2)
# df1.columns = Feature_all
# Num_seg = max(df1['segment'])
# threshold_time = 120
# for i in range(Num_seg+1):
#     df_temp = df1[df1['segment']==i]
#     range_time = df_temp.iloc[-1]['index'] - df_temp.iloc[0]['index']
#     range_len = len(df_temp)
#     if (range_time<threshold_time) | (range_len<0.8*range_time):
#         df1 = df1.drop(df1[df1['segment']==i].index)

print('删除时间异常片段完成')
time_end = time.time()
print('删除时间异常片段用时%d秒' %(time_end-time_start))
print('数据总样本为%d条' %(len(df1)))

# 片段重编号
time_start = time.time()
print('正在片段重编号')

df1 = df1.reset_index(drop=True)
df1['segment'] = 0
seg = 0
length1 = len(df1)
for i in np.arange(1,length1):
    if df1['index'][i]-df1['index'][i-1] > threshold_seg:
        seg = seg + 1
        df1.loc[i, 'segment'] = seg
    else:
        df1.loc[i, 'segment'] = seg

# array2 = np.array(df1)
# seg = 0
# length1 = array2.shape[0]
# for i in np.arange(1,length1):
#     if array2[i,dic_fea['index']]-array2[i-1,dic_fea['index']] > threshold_seg:
#         seg = seg + 1
#         array2[i,dic_fea['segment']] = seg
#     else:
#         array2[i,dic_fea['segment']] = seg

print('片段重编号完成')
time_end = time.time()
print('片段重编号用时%d秒' %(time_end-time_start))

# 线性插值补齐缺失数据
time_start = time.time()
print('正在线性插值补齐缺失数据')

Feature_inter = ['GPS经度','GPS纬度','GPS车速','GPS方向','整车状态','充电状态','动力电池总电压','动力电池总电流',
                'SOC','油门踏板状态','制动踏板状态','驱动电机转速','驱动电机扭矩','road_level','bus_stop','inter','slope_real','hight']
df2 = pd.DataFrame()
Num_seg = max(df1['segment'])
for i in range(Num_seg+1):
    df_temp2 = pd.DataFrame()
    df_temp = df1[df1['segment']==i]
    x = df_temp['index']
    xx = np.arange(df_temp.iloc[0]['index'], df_temp.iloc[-1]['index']+1)
    df_temp2['index']=xx
    df_temp2['formate_time']=pd.to_datetime(df_temp2['index'],unit='s')
    df_temp2['segment']=i
    for feature in Feature_inter:
        y = df_temp[feature]
        y_min = min(y)
        y_max = max(y)
        f = interpolate.interp1d(x,y,kind ='linear')
        yy = f(xx)
        df_temp2[feature] = yy
    df2 = df2._append(df_temp2, ignore_index=True)
df2['road_level'] = df2['road_level'].round()
df2['bus_stop'] = df2['bus_stop'].round()
df2['inter'] = df2['inter'].round()
df2.loc[df2['GPS车速']<0.5, 'GPS车速']=0
df2.loc[df2['油门踏板状态']<0.5, '油门踏板状态']=0
df2.loc[df2['制动踏板状态']<0.5, '制动踏板状态']=0
df2.loc[df2['驱动电机转速']<0.5, '驱动电机转速']=0
df2.loc[(df2['驱动电机扭矩']<0.5) & (df2['驱动电机扭矩']>-0.5), '驱动电机扭矩']=0
df2.loc[df2['GPS方向']<0.5, 'GPS方向']=0

print('线性插值补齐缺失数据已完成')
time_end = time.time()
print('线性插值补齐缺失数据用时%d秒' %(time_end-time_start))
print('数据总样本为%d条' %(len(df2)))

time_start = time.time()
print('正在计算加速度')
df2['加速度'] = 0
for i in range(len(df2)-1):
    if df2['segment'][i] == df2['segment'][i+1]:
        df2.loc[i,'加速度'] = (df2['GPS车速'][i+1] - df2['GPS车速'][i])/3.6
    else:
        df2.loc[i,'加速度'] = np.nan
df2.loc[len(df2)-1,'加速度'] = np.nan
df2 = df2.dropna(how='any')
df2 = df2.reset_index(drop=True)
print('计算加速度已完成')
time_end = time.time()
print('计算加速度用时%d秒' %(time_end-time_start))

time_start = time.time()
print('正在删除异常加速度')
acc_out_after = df2[np.absolute(df2['加速度'])>=4]['index']
acc_out_before = acc_out_after - 1
acc_out = np.union1d(acc_out_after, acc_out_before)
for i in range(len(df2)):
    if df2['index'][i] in acc_out:
        df2.loc[i, '加速度'] = np.nan
df1 = df2.dropna(how='any') # 删除缺失数据
print('删除异常加速度已完成')
time_end = time.time()
print('删除异常加速度用时%d秒' %(time_end-time_start))

# 片段重编号
time_start = time.time()
print('正在片段重编号')

df1 = df1.reset_index(drop=True)
df1['segment'] = 0
seg = 0
length1 = len(df1)
for i in np.arange(1,length1):
    if df1['index'][i]-df1['index'][i-1] > threshold_seg:
        seg = seg + 1
        df1.loc[i, 'segment'] = seg
    else:
        df1.loc[i, 'segment'] = seg

print('片段重编号完成')
time_end = time.time()
print('片段重编号用时%d秒' %(time_end-time_start))

# 线性插值补齐缺失数据
time_start = time.time()
print('正在线性插值补齐缺失数据')
Feature_inter2 = ['GPS经度','GPS纬度','GPS车速','GPS方向','整车状态','充电状态','动力电池总电压','动力电池总电流',
                'SOC','油门踏板状态','制动踏板状态','驱动电机转速','驱动电机扭矩','road_level','bus_stop','inter','slope_real','hight','加速度']
df2 = pd.DataFrame()
Num_seg = max(df1['segment'])
for i in range(Num_seg+1):
    df_temp2 = pd.DataFrame()
    df_temp = df1[df1['segment']==i]
    x = df_temp['index']
    xx = np.arange(df_temp.iloc[0]['index'], df_temp.iloc[-1]['index']+1)
    df_temp2['index']=xx
    df_temp2['formate_time']=pd.to_datetime(df_temp2['index'],unit='s')
    df_temp2['segment']=i
    for feature in Feature_inter2:
        y = df_temp[feature]
        y_min = min(y)
        y_max = max(y)
        f = interpolate.interp1d(x,y,kind ='linear')
        yy = f(xx)
        df_temp2[feature] = yy
    df2 = df2._append(df_temp2, ignore_index=True)
df2['road_level'] = df2['road_level'].round()
df2['bus_stop'] = df2['bus_stop'].round()
df2['inter'] = df2['inter'].round()
df2.loc[df2['GPS车速']<0.5, 'GPS车速']=0
df2.loc[df2['油门踏板状态']<0.5, '油门踏板状态']=0
df2.loc[df2['制动踏板状态']<0.5, '制动踏板状态']=0
df2.loc[df2['驱动电机转速']<0.5, '驱动电机转速']=0
df2.loc[(df2['驱动电机扭矩']<0.5) & (df2['驱动电机扭矩']>-0.5), '驱动电机扭矩']=0
df2.loc[df2['GPS方向']<0.5, 'GPS方向']=0

print('线性插值补齐缺失数据已完成')
time_end = time.time()
print('线性插值补齐缺失数据用时%d秒' %(time_end-time_start))
print('数据总样本为%d条' %(len(df2)))

# 计算能耗
df2['ECR'] = df2['动力电池总电压']*df2['动力电池总电流']/1000

# 日期换算
df2['month'] = df2['formate_time'].dt.month
df2['day'] = df2['formate_time'].dt.day
df2['hour'] = df2['formate_time'].dt.hour

# 融合交通流状况
time_start = time.time()
print('正在融合交通数据')
# peak hour
df2['peak']=0
df2.loc[(df2['hour']>=7) & (df2['hour']<9), 'peak'] = 1
df2.loc[(df2['hour']>=17) & (df2['hour']<19), 'peak'] = 1
# weekend
df2['weekend']=0
df2.loc[(df2['month']==4) & (df2['day']==24), 'weekend'] = 1
df2.loc[(df2['month']==4) & (df2['day']==25), 'weekend'] = 1
df2.loc[(df2['month']==5) & (df2['day']==1), 'weekend'] = 1
df2.loc[(df2['month']==5) & (df2['day']==2), 'weekend'] = 1
df2.loc[(df2['month']==5) & (df2['day']==8), 'weekend'] = 1
df2.loc[(df2['month']==5) & (df2['day']==9), 'weekend'] = 1
print('融合交通数据已完成')
time_end = time.time()
print('融合交通数据用时%d秒' %(time_end-time_start))

# 融合天气状况
time_start = time.time()
print('正在融合天气数据')

# df_w = pd.read_csv(path_data + '//weather.csv')
# df2['T'] = 0
# df2['P'] = 0
# df2['U'] = 0
# df2['VV'] = 0
# df2['WW'] = 0
# for i in range(len(df2)):
#     month = df2['month'][i]
#     day = df2['day'][i]
#     hour = df2['hour'][i]
#     df_temp3 = df_w[(df_w['month']==month) & (df_w['day']==day) & (df_w['hour']==hour)]
#     df2.loc[i, 'T'] = df_temp3.iloc[0, 1]
#     df2.loc[i, 'P'] = df_temp3.iloc[0, 2]
#     df2.loc[i, 'U'] = df_temp3.iloc[0, 3]
#     df2.loc[i, 'VV'] = df_temp3.iloc[0, 4]
#     df2.loc[i, 'WW'] = df_temp3.iloc[0, 5]

df_w = pd.read_csv(path_data + '//weather.csv')
df2['T'] = 0
df2['P'] = 0
df2['U'] = 0
df2['VV'] = 0
df2['WW'] = 0
Feature_weather = ['month', 'day', 'hour', 'peak', 'weekend', 'T', 'P', 'U', 'VV', 'WW']
df_weather = df2[Feature_weather]
array_weather = np.array(df_weather)
for i in range(len(df_weather)):
    month = array_weather[i,0]
    day = array_weather[i,1]
    hour = array_weather[i,2]
    df_temp3 = df_w[(df_w['month']==month) & (df_w['day']==day) & (df_w['hour']==hour)]
    array3 = np.array(df_temp3)
    array_weather[i,5] = array3[0, 1]
    array_weather[i,6] = array3[0, 2]
    array_weather[i,7] = array3[0, 3]
    array_weather[i,8] = array3[0, 4]
    array_weather[i,9] = array3[0, 5]
df2.loc[:,Feature_weather] = array_weather

print('融合天气数据已完成')
time_end = time.time()
print('融合天气数据用时%d秒' %(time_end-time_start))

df2.columns = ['index', 'for_tim', 'seg', 'lon', 'lat', 'vel', 'dir_GPS', 'veh_sta', 'cha', 'vol', 'cur', 
                'SOC', 'gas_sta', 'bra_sta', 'v_rot', 'tor','road_level','bus_stop','inter','slope_real','hight', 'acc', 'ECR', 
                'month', 'day', 'hour', 'peak', 'weekend', 
                'T', 'P', 'U', 'VV', 'WW']

df2.loc[df2['vel']<0.5, 'vel']=0
df2.loc[df2['gas_sta']<0.5, 'gas_sta']=0
df2.loc[df2['bra_sta']<0.5, 'bra_sta']=0
df2.loc[df2['v_rot']<0.5, 'v_rot']=0
df2.loc[(df2['tor']<0.5) & (df2['tor']>-0.5), 'tor']=0
df2.loc[df2['dir_GPS']<0.5, 'dir_GPS']=0

# 片段重编号
time_start = time.time()
print('正在片段重编号')

df2 = df2.reset_index(drop=True)
df2['seg'] = 0
seg = 0
length1 = len(df2)
for i in np.arange(1,length1):
    if df2['index'][i]-df2['index'][i-1] > threshold_seg:
        seg = seg + 1
        df2.loc[i, 'seg'] = seg
    else:
        df2.loc[i, 'seg'] = seg

print('片段重编号完成')
time_end = time.time()
print('片段重编号用时%d秒' %(time_end-time_start))

# 计算两点间距离
time_start = time.time()
print('正在计算距离和高差')

df2['dis'] = 0
df2['ele'] = 0
Len = list()
Ran = list()
df3 = pd.DataFrame()
Num_seg = max(df2['seg'])
for i in range(Num_seg+1):
    df_temp = df2[df2['seg']==i]
    df_temp = df_temp.reset_index(drop=True)  
    Len.append(len(df_temp))
    Ran.append(df_temp.iloc[-1]['index']-df_temp.iloc[0]['index']+1)
    for j in range(1,len(df_temp)):
        dis = Geodesic.WGS84.Inverse(df_temp['lat'][j], df_temp['lon'][j],
                                        df_temp['lat'][j-1], df_temp['lon'][j-1])['s12']
        if dis > 15:
            dis = df_temp['vel'][j]/3.6
        df_temp.loc[j,'dis'] = dis
        df_temp.loc[j,'ele'] = df_temp['hight'][j] - df_temp['hight'][j-1]
    df3 = df3._append(df_temp, ignore_index=True)

print('计算距离和高差已完成')
time_end = time.time()
print('计算距离和高差用时%d秒' %(time_end-time_start))

describe = df3.describe()
df3.to_csv(path_data + '\\road_30.csv',index=False)