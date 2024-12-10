import pandas as pd
import numpy as np
from geographiclib.geodesic import Geodesic

path_data = r''
df = pd.read_csv(path_data + '\\data_remove.csv')
df_point = pd.read_csv(path_data + '\\data_point.csv')
df_road = pd.read_csv(path_data + '\\data_road.csv')
df_hight = pd.read_csv(path_data + '\\data_hight.csv')
df_aspect = pd.read_csv(path_data + '\\data_aspect.csv')
df_slope = pd.read_csv(path_data + '\\data_slope.csv')

df_aspect = df_aspect.drop_duplicates(subset=['index','GPS经度','GPS纬度','动力电','动力_1'], keep='first')
df_hight = df_hight.drop_duplicates(subset=['index','GPS经度','GPS纬度','动力电','动力_1'], keep='first')
df_slope = df_slope.drop_duplicates(subset=['index','GPS经度','GPS纬度','动力电','动力_1'], keep='first')

df['road_level'] = df_road['type']
df['bottleneck'] = df_point['NEAR_FC']
df['bot_dist'] = df_point['NEAR_DIST']
df['road_id'] = df_road['name']
df.loc[(df['road_level']=='motorway') | (df['road_level']=='motorway_link'), 'road_level'] = 4
df.loc[(df['road_level']=='trunk') | (df['road_level']=='trunk_link'), 'road_level'] = 3
df.loc[(df['road_level']=='primary') | (df['road_level']=='primary_link'), 'road_level'] = 3
df.loc[(df['road_level']=='secondary') | (df['road_level']=='secondary_link'), 'road_level'] = 2
df.loc[(df['road_level']=='tertiary') | (df['road_level']=='tertiary_link'), 'road_level'] = 2
df.loc[(df['road_level']!=2) & (df['road_level']!=3) & (df['road_level']!=4), 'road_level'] = 1

df['signal'] = 0
df['crossing'] = 0
df['bus_stop'] = 0
df['park_ent'] = 0
df['inter'] = 0
df.loc[df['bottleneck']=='signal', 'signal'] = 1
df.loc[df['bottleneck']=='crossing', 'crossing'] = 1
df.loc[(df['bottleneck']=='stop') & (df['bot_dist']<0.0002), 'bus_stop'] = 1
df.loc[(df['bottleneck']=='entrance') & (df['bot_dist']<0.0002), 'park_ent'] = 1
df.loc[(df['bottleneck']=='crossing') | (df['bottleneck']=='signal'), 'inter'] = 1

df['bottleneck'] = df['signal'] + df['crossing'] + df['bus_stop'] + df['park_ent']

df['aspect'] = df_aspect['RASTERVALU']
df['slope'] = df_slope['RASTERVALU']
df['angle'] = np.absolute(df['aspect'] - df['GPS方向'])
df['slope_real'] = df['slope'] * np.cos(df['angle']*np.pi/180)

df['hight'] = df_hight['RASTERVALU']

# df['vis'] = 0
# df.loc[df['VV']>1, 'vis'] = 4
# df.loc[df['VV']<=1, 'vis'] = 3
# df.loc[df['VV']<=0.5, 'vis'] = 2
# df.loc[df['VV']<=0.2, 'vis'] = 1

# Calculate the distance
# df['hight'] = df['RASTERVALU']
# df['dis'] = 0
# df['ele'] = 0
# Len = list()
# Ran = list()
# df1 = pd.DataFrame()
# Num_seg = max(df['seg'])
# for i in range(Num_seg+1):
#     df_temp = df[df['seg']==i]
#     df_temp = df_temp.reset_index(drop=True)  
#     Len.append(len(df_temp))
#     Ran.append(df_temp.iloc[-1]['index']-df_temp.iloc[0]['index']+1)
#     for j in range(1,len(df_temp)):
#         dis = Geodesic.WGS84.Inverse(df_temp['lat'][j], df_temp['lon'][j],
#                                         df_temp['lat'][j-1], df_temp['lon'][j-1])['s12']
#         if dis > 15:
#             dis = df_temp['vel'][j]/3.6
#         df_temp.loc[j,'dis'] = dis
#         df_temp.loc[j,'ele'] = df_temp['hight'][j] - df_temp['hight'][j-1]
        
#     df1 = df1._append(df_temp, ignore_index=True)

des = df.describe()
df.to_csv(path_data + '\\data_fus.csv',index=False)