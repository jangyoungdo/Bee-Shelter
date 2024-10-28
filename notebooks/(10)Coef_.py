import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 꿀벌과 말벌 데이터셋을 불러옵니다
df_mfcc_bee = pd.read_csv('/content/drive/MyDrive/sound/CSV/MFCC_CSV/mfcc_honeybee.csv')
df_mfcc_hornet = pd.read_csv('/content/drive/MyDrive/sound/CSV/MFCC_CSV/mfcc_hornet.csv')

# n_samples를 100개로 고정 (꿀벌과 말벌 각각)
n_samples = 100

# 꿀벌 및 말벌 데이터에서 각각 n_samples 만큼 샘플링
df_mfcc_bee_sampled = df_mfcc_bee.sample(n=n_samples, random_state=42)
df_mfcc_hornet_sampled = df_mfcc_hornet.sample(n=n_samples, random_state=42)

# 라벨 추가
df_mfcc_bee_sampled['Label'] = 0  # 꿀벌은 0
df_mfcc_hornet_sampled['Label'] = 1  # 말벌은 1

# 데이터 결합
df_combined = pd.concat([df_mfcc_bee_sampled, df_mfcc_hornet_sampled], axis=0)

# Train, Test, Validation Set 나누기
X = df_combined.drop(columns=['Label'])
y = df_combined['Label']

# Train (60%), Validation (20%), Test (20%) 세트로 분할
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 로지스틱 회귀 모델 학습
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)

# 훈련 데이터에 대한 정확도 계산
y_train_pred = logreg.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# 검증 데이터에 대한 정확도 계산
y_val_pred = logreg.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

# 테스트 데이터에 대한 정확도 계산
y_test_pred = logreg.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# 출력
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 로지스틱 회귀의 회귀 계수 확인
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]
feature_names = X.columns

# 가중치와 독립 변수 이름을 함께 확인하기 위한 데이터프레임 생성
df_coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# 가중치의 절댓값이 큰 순서대로 중요도 정렬
df_coefficients['Importance'] = np.abs(df_coefficients['Coefficient'])
df_coefficients = df_coefficients.sort_values(by='Importance', ascending=False)

# 가장 중요한 독립변수 출력
print("\nTop 5 Most Important Features based on Logistic Regression Coefficients:")
print(df_coefficients.head())

# 변수 중요도 시각화
plt.figure(figsize=(10, 6))
plt.barh(df_coefficients['Feature'], df_coefficients['Importance'], color='blue')
plt.xlabel('Importance (Absolute Coefficient Value)')
plt.ylabel('Features')
plt.title('Feature Importance in Logistic Regression')
plt.gca().invert_yaxis()
plt.show()

# 가중치와 절편 (bias) 출력
print("\nFull Model Coefficients and Bias:")
print(f"Intercept (Bias): {intercept}")
print(df_coefficients)

# 신뢰도 기준 설정
confidence_threshold = 0.95

# 지도 학습에서 사용되지 않은 데이터 샘플링
unlabeled_hornet = df_mfcc_hornet.drop(df_mfcc_hornet_sampled.index)
unlabeled_bee = df_mfcc_bee.drop(df_mfcc_bee_sampled.index)

# unlabeled 데이터를 결합
unlabeled_data = pd.concat([unlabeled_hornet, unlabeled_bee], axis=0)

# 예측 확률 계산 및 신뢰도 높은 데이터 선택
X_unlabeled = unlabeled_data.drop(columns=['Label'], errors='ignore')
probs = logreg.predict_proba(X_unlabeled)
high_confidence_mask = np.max(probs, axis=1) > confidence_threshold

# 신뢰도 높은 데이터만 pseudo-labeled로 추가
pseudo_labeled_data = unlabeled_data[high_confidence_mask].copy()
y_unlabeled_pred = logreg.predict(X_unlabeled[high_confidence_mask])
pseudo_labeled_data['Label'] = y_unlabeled_pred

# Pseudo-labeled 데이터와 기존 데이터를 결합하여 다시 학습
df_combined = pd.concat([df_combined, pseudo_labeled_data], ignore_index=True)

# 다시 학습을 위한 데이터 준비
X = df_combined.drop(columns=['Label'])
y = df_combined['Label']

# Train Set과 Validation Set 재구성
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 로지스틱 회귀 모델 재학습
logreg.fit(X_train, y_train)

# Pseudo Labeling 후 테스트 데이터에 대한 정확도 계산
y_test_pred = logreg.predict(X_test)
pseudo_test_accuracy = accuracy_score(y_test, y_test_pred)

# 수도 라벨링 적용 후 성능 출력
print(f"Test Accuracy with Pseudo Labeling: {pseudo_test_accuracy * 100:.2f}%")
