import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 꿀벌과 말벌 데이터셋을 불러옵니다
df_mfcc_bee = pd.read_csv('/content/drive/MyDrive/sound/CSV/MFCC_CSV/mfcc_honeybee.csv')
df_mfcc_hornet = pd.read_csv('/content/drive/MyDrive/sound/CSV/MFCC_CSV/mfcc_hornet.csv')

# n_samples를 100개로 고정
n_samples = 100

# 꿀벌 및 말벌 데이터에서 각각 n_samples 만큼 샘플링
df_mfcc_bee_sampled = df_mfcc_bee.sample(n=n_samples, random_state=42)
df_mfcc_hornet_sampled = df_mfcc_hornet.sample(n=n_samples, random_state=42)

# 라벨 추가 ('B'는 꿀벌, 'H'는 말벌을 나타냄)
df_mfcc_bee_sampled['Label'] = 'B'
df_mfcc_hornet_sampled['Label'] = 'H'

# 데이터 결합
df_combined = pd.concat([df_mfcc_bee_sampled, df_mfcc_hornet_sampled], axis=0)

# 'Label' 열을 제외하고 상관관계 확인
df_mfcc_features = df_combined.drop(columns=['Label'], errors='ignore')  # 'Label' 열을 안전하게 제거

# 상관관계 행렬 계산
correlation_matrix = df_mfcc_features.corr()

# 상관관계 행렬 시각화 (heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of MFCC Features')
plt.show()
