# 필요 라이브러리 임포트
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 데이터 준비 및 라벨 추가
# 데이터프레임에 라벨 추가
df_mfcc_bee['Label'] = 'B'
df_mfcc_hornet['Label'] = 'H'

# 데이터프레임 결합
df_combined = pd.concat([df_mfcc_bee, df_mfcc_hornet], axis=0)

# 라벨 인코딩
y = df_combined['Label']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 2. 데이터 스케일링 및 학습/검증/테스트 세트 나누기
# MFCC 피처 설정
X = df_combined.drop(columns=['Label'])

# 데이터 스케일링 (정규화)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train (60%), Validation (20%), Test (20%) 세트로 분할
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. K-겹 교차 검증을 통한 모델 평가
kf = KFold(n_splits=5, shuffle=True, random_state=42)
logreg = LogisticRegression(solver='liblinear')
accuracies = []

for train_index, test_index in kf.split(X_train):
    X_kf_train, X_kf_val = X_train[train_index], X_train[test_index]
    y_kf_train, y_kf_val = y_train[train_index], y_train[test_index]

    # 모델 학습
    logreg.fit(X_kf_train, y_kf_train)

    # 검증 데이터에 대한 정확도 계산
    accuracy = logreg.score(X_kf_val, y_kf_val)
    accuracies.append(accuracy)

print(f"K-겹 교차 검증 정확도: {np.mean(accuracies) * 100:.2f}% ± {np.std(accuracies) * 100:.2f}%")

# 4. 모델 학습 및 학습률 추이 확인
epochs = 50
train_accuracies = []
learning_rates = []

for epoch in range(1, epochs + 1):
    logreg = LogisticRegression(solver='liblinear', max_iter=epoch)
    logreg.fit(X_train, y_train)

    # 학습 정확도 기록
    train_accuracy = logreg.score(X_train, y_train)
    train_accuracies.append(train_accuracy)

    # 학습률 기록
    learning_rates.append(logreg.coef_.mean())

# 학습 정확도 및 학습률 추이 시각화
plt.figure(figsize=(14, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(range(1, epochs + 1), learning_rates, label='Learning Rate', color='g')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate over Epochs')
plt.grid()
plt.legend()
plt.show()

# 5. 테스트 세트에 대한 최종 평가
y_test_pred = logreg.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")
