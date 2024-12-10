import pandas as pd
import numpy as np

path_data = r''
df = pd.read_csv(path_data + '\\data.csv')

# Num_sample = len(df)
# Num_seg = max(df['seg'])
# Dis = []
# Sample = []
# for i in range(Num_seg+1):
#     df_temp = df[df['seg']==i]
#     dis_temp = np.sum(df_temp['dis'])
#     sample_temp = max(df_temp['index']) - min(df_temp['index']) + 1
#     Dis.append(dis_temp)
#     Sample.append(sample_temp)
# Num_sample_conf = sum(Sample)
# Dis = np.array(Dis)
# Sample = np.array(Sample)

# print('样本数量为%d' %Num_sample)
# print('行程数量为%d' %Num_seg)
# print('行程最大样本量为%d' %np.max(Sample))
# print('行程最小样本量为%d' %np.min(Sample))
# print('行程样本量中位数为%d' %np.median(Sample))
# print('最大行程距离为%d' %np.max(Dis))
# print('最小行程距离为%d' %np.min(Dis))
# print('行程距离中位数为%d' %np.median(Dis))

# Result = []
# Result.append(Num_sample)
# Result.append(Num_seg)
# Result.append(np.max(Sample))
# Result.append(np.min(Sample))
# Result.append(np.median(Sample))
# Result.append(np.max(Dis))
# Result.append(np.min(Dis))
# Result.append(np.median(Dis))
# Result = np.array(Result).reshape(1,-1)

gap = 1000
Num_seg = max(df['seg'])
Seg = []
Num = []
for i in range(Num_seg+1):
    df_temp = df[df['seg']==i]
    df_temp = df_temp.reset_index(drop=True)
    j = 0
    while (j+1)*gap < len(df_temp):
        ecr = np.sum(df_temp['ECR'][j*gap:(j+1)*gap])
        if ecr < 0:
            Seg.append(i)
            Num.append(j)
        j += 1

df_ECR = df[df['seg']==38]
df_ECR = df_ECR.reset_index(drop=True)
df_ECR = df_ECR.loc[1*gap:2*gap,['SOC', 'ECR']]
