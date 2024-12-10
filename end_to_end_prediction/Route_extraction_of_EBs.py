import pandas as pd
import numpy as np

path_data = r''
df2 = pd.read_csv(path_data + '\\09_16_19_fus.csv')
df4 = pd.read_csv(path_data + '\\21_fus.csv')
df5 = pd.read_csv(path_data + '\\22_fus.csv')
df6 = pd.read_csv(path_data + '\\26_fus.csv')
df7 = pd.read_csv(path_data + '\\28_fus.csv')
df8 = pd.read_csv(path_data + '\\38_fus.csv')
df9 = pd.read_csv(path_data + '\\54_fus.csv')

df = pd.concat([df2, df4, df5, df6, df7, df8, df9],ignore_index=True)
df['ECR'] = df['ECR']/3600

Seg = []
Seg.append(max(df2['seg']))
Seg.append(max(df4['seg']))
Seg.append(max(df5['seg']))
Seg.append(max(df6['seg']))
Seg.append(max(df7['seg']))
Seg.append(max(df8['seg']))
Seg.append(max(df9['seg']))

df['seg_reset'] = 0
seg = 0
length1 = len(df)
for i in np.arange(1,length1):
    if df['seg'][i] !=  df['seg'][i-1]:
        seg = seg + 1
        df.loc[i, 'seg_reset'] = seg
    else:
        df.loc[i, 'seg_reset'] = seg

df['seg'] = df['seg_reset']

df_road = pd.DataFrame()
Num_seg = max(df['seg'])
for i in range(Num_seg+1):
    df_temp = df[df['seg']==i]
    df_cho = df_temp[(df_temp['road_id'] == '麒麟门大道') | (df_temp['road_id'] == '中山门大街') | (df_temp['road_id'] == '白水桥')]
    
    # df_cho = df_temp[(df_temp['road_id'] == '沪蓉高速') | (df_temp['road_id'] == '牌接北路') | (df_temp['road_id'] == '紫金东路')
    #                   | (df_temp['road_id'] == '东苑路') | (df_temp['road_id'] == '后标营路') | (df_temp['road_id'] == '后标营街')]
    
    # df_cho = df_temp[(df_temp['road_id'] == '御道街') | (df_temp['road_id'] == '大光路') | (df_temp['road_id'] == '光华路')
    #                   | (df_temp['road_id'] == '象房村路') | (df_temp['road_id'] == '解放南路') | (df_temp['road_id'] == '大中桥')
    #                   | (df_temp['road_id'] == '建康路') | (df_temp['road_id'] == '太平南路') | (df_temp['road_id'] == '长白街')
    #                   | (df_temp['road_id'] == '白下路') | (df_temp['road_id'] == '中山南路') | (df_temp['road_id'] == '升州路')
    #                   | (df_temp['road_id'] == '莫愁路')]
    
    # df_cho = df_temp[(df_temp['road_id'] == '北京东路') | (df_temp['road_id'] == '中山北路') | (df_temp['road_id'] == '太平北路')
    #                   | (df_temp['road_id'] == '鼓楼广场') | (df_temp['road_id'] == '光华路') | (df_temp['road_id'] == '黄埔路')
    #                   | (df_temp['road_id'] == '御道街') | (df_temp['road_id'] == '苜蓿园大') | (df_temp['road_id'] == '解放路')
    #                   | (df_temp['road_id'] == '瑞金路') | (df_temp['road_id'] == '珠江路')]
    
    # df_cho = df_temp[(df_temp['road_id'] == '仙林大道') | (df_temp['road_id'] == '环陵路') | (df_temp['road_id'] == '马群南路')
    #                   | (df_temp['road_id'] == '紫东路')]
    
    df_cho = df_cho.reset_index(drop=True)
    if len(df_cho) > 5:
        start = df_cho['index'][0]
        for j in range(1,len(df_cho)):
            if df_cho['index'][j]-df_cho['index'][j-1]>300:
                stop = df_cho['index'][j-1]
                df_choose = df_temp[(df_temp['index'] >= start) & (df_temp['index'] <= stop)]
                df_road = pd.concat([df_road, df_choose],ignore_index=True)
                start = df_cho['index'][j]
        stop = df_cho['index'][len(df_cho)-1]
        df_choose = df_temp[(df_temp['index'] >= start) & (df_temp['index'] <= stop)]
        df_road = pd.concat([df_road, df_choose],ignore_index=True)

df_road['seg'] = 0
seg = 0
length1 = len(df_road)
threshold_seg = 300
for i in np.arange(1,length1):
    if np.absolute(df_road['index'][i]-df_road['index'][i-1]) > threshold_seg:
        seg = seg + 1
        df_road.loc[i, 'seg'] = seg
    else:
        df_road.loc[i, 'seg'] = seg

Num_seg = max(df_road['seg'])
threshold_time = 300
for i in range(Num_seg+1):
    df_temp = df_road[df_road['seg']==i]
    range_len = len(df_temp)
    if range_len < threshold_time:
        df_road = df_road.drop(df_road[df_road['seg']==i].index)
df_road = df_road.reset_index(drop=True)  

df_road['seg'] = 0
seg = 0
length1 = len(df_road)
threshold_seg = 300
for i in np.arange(1,length1):
    if np.absolute(df_road['index'][i]-df_road['index'][i-1]) > threshold_seg:
        seg = seg + 1
        df_road.loc[i, 'seg'] = seg
    else:
        df_road.loc[i, 'seg'] = seg

df[['slope_max', 'slope_min', 'vel_std', 'acc_std', 'gas_std', 'bra_std', 'rot_std', 'tor_std']] = 0

feature = ['SOC', 'T', 'P', 'U', 'WW',
            'peak', 'weekend', 'bus_stop', 'inter', 'road_level',
            'ele', 'slope_real', 'slope_max', 'slope_min', 'vel', 
            'acc', 'gas_sta', 'bra_sta', 'v_rot', 'tor',
            'vel_std', 'acc_std', 'gas_std', 'bra_std', 'rot_std', 
            'tor_std', 'ECR', 'dis', 'seg']

df1 = df[feature]
description = df1.describe()

df1.to_csv(path_data + '\\data.csv',index=False)