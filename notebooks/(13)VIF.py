# n_samples를 100개로 고정
n_samples = 100

# 꿀벌 및 말벌 데이터에서 각각 n_samples 만큼 샘플링
df_mfcc_bee_sampled = df_mfcc_bee.sample(n=n_samples, random_state=42)
df_mfcc_hornet_sampled = df_mfcc_hornet.sample(n=n_samples, random_state=42)

# 라벨 추가
df_mfcc_bee_sampled['Label'] = 'B'
df_mfcc_hornet_sampled['Label'] = 'H'

# 데이터 결합 (라벨을 제외한 MFCC 특징 사용)
df_combined = pd.concat([df_mfcc_bee_sampled, df_mfcc_hornet_sampled], axis=0).reset_index(drop=True)

# 데이터프레임의 열 이름 확인
print(df_combined.columns)

# 라벨을 제거하고 VIF 계산을 위한 독립 변수만 추출
X = df_combined.drop(columns=['Label'], errors='ignore')  # 'Label'이 없을 경우에도 에러 방지

# VIF 값을 계산하는 함수
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns  # 각 독립변수 이름
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# VIF 값 계산
vif_result = calculate_vif(X)

# VIF 값 출력
print(vif_result)
