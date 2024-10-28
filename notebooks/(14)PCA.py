
# n_samples를 100개로 고정
n_samples = 100

# 꿀벌 및 말벌 데이터에서 각각 n_samples 만큼 샘플링
df_mfcc_bee_sampled = df_mfcc_bee.sample(n=n_samples, random_state=42)
df_mfcc_hornet_sampled = df_mfcc_hornet.sample(n=n_samples, random_state=42)

# 라벨 추가
df_mfcc_bee_sampled['Label'] = 'B'
df_mfcc_hornet_sampled['Label'] = 'H'

# 데이터 결합
df_combined = pd.concat([df_mfcc_bee_sampled, df_mfcc_hornet_sampled], axis=0).reset_index(drop=True)

# Label 열을 제외한 모든 MFCC 특징을 사용하여 PCA 적용
X = df_combined.drop(columns=['Label'])
y = df_combined['Label']

# PCA 적용 (주성분 5개로 변환)
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X)

# PCA 적용한 결과를 데이터프레임으로 변환
df_combined_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(7)])
df_combined_pca['Label'] = y.reset_index(drop=True)

# Label을 수치형으로 변환
y_encoded = pd.get_dummies(df_combined_pca['Label']).values.argmax(axis=1)

# Train (60%), Validation (20%), Test (20%) 세트로 분할
X_train, X_temp, y_train, y_temp = train_test_split(X_pca, y_encoded, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# NaN 값 처리 (평균 대체 또는 NaN 값 제거 가능)
X_train = pd.DataFrame(X_train).fillna(X_train.mean())  # NaN 값을 각 열의 평균으로 대체
X_val = pd.DataFrame(X_val).fillna(X_val.mean())
X_test = pd.DataFrame(X_test).fillna(X_test.mean())

# 로지스틱 회귀 모델 학습
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)

# 훈련 데이터에 대한 정확도 계산
y_train_pred = logreg.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# 검증 데이터에 대한 예측
y_val_pred = logreg.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

# 테스트 데이터 예측
y_test_pred = logreg.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# 수도 레이블링 과정
# 지도 학습에서 사용되지 않은 데이터 (unlabeled_hornet)
unlabeled_hornet = df_mfcc_hornet.drop(df_mfcc_hornet_sampled.index).reset_index(drop=True)

# PCA 적용 (주성분 5개로 변환)
unlabeled_hornet_pca = pca.transform(unlabeled_hornet)

# NaN 값 처리 (평균 대체 또는 NaN 값 제거 가능)
unlabeled_hornet_pca = pd.DataFrame(unlabeled_hornet_pca).fillna(unlabeled_hornet_pca.mean())

# 예측된 레이블을 기준으로 신뢰도 높은 데이터 선택
probs = logreg.predict_proba(unlabeled_hornet_pca)
confidence_threshold = 0.95
high_confidence_mask = np.max(probs, axis=1) > confidence_threshold  # 신뢰도 기준

# 신뢰도 높은 데이터만 pseudo-labeled로 추가
pseudo_labeled_data = unlabeled_hornet[high_confidence_mask].copy()
y_unlabeled_pred = logreg.predict(unlabeled_hornet_pca)
pseudo_labeled_data['Label'] = y_unlabeled_pred[high_confidence_mask]

# Pseudo-labeled 데이터와 기존 데이터를 결합하여 다시 학습 (인덱스 재설정)
df_pseudo_combined = pd.concat([df_combined_pca, pseudo_labeled_data], ignore_index=True).reset_index(drop=True)

# 다시 학습을 위한 데이터 준비
X_pseudo = df_pseudo_combined.drop(columns=['Label'])
y_pseudo = pd.get_dummies(df_pseudo_combined['Label']).values.argmax(axis=1)

# NaN 값 처리 (평균 대체 또는 NaN 값 제거 가능)
X_pseudo = pd.DataFrame(X_pseudo).fillna(X_pseudo.mean())

# Train Set과 Validation Set 재구성
X_train_pseudo, X_temp_pseudo, y_train_pseudo, y_temp_pseudo = train_test_split(X_pseudo, y_pseudo, test_size=0.4, random_state=42)
X_val_pseudo, X_test_pseudo, y_val_pseudo, y_test_pseudo = train_test_split(X_temp_pseudo, y_temp_pseudo, test_size=0.5, random_state=42)

# 로지스틱 회귀 모델 재학습
logreg_pseudo = LogisticRegression(solver='liblinear')
logreg_pseudo.fit(X_train_pseudo, y_train_pseudo)

# Pseudo Labeling 후 테스트 데이터에 대한 예측
y_test_pseudo_pred = logreg_pseudo.predict(X_test_pseudo)
pseudo_test_accuracy = accuracy_score(y_test_pseudo, y_test_pseudo_pred)

# 결과 출력
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Accuracy with Pseudo Labeling: {pseudo_test_accuracy * 100:.2f}%")

# 1. PCA 설명된 분산 비율 시각화
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center',
        label='Individual explained variance')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('PCA Explained Variance Ratio')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# PCA 구성 요소 확인 (각 주성분이 원래 특징들로부터 어떻게 구성되었는지)
# 각 주성분에 대한 원래 특징들의 가중치 확인
pca_components = pca.components_
feature_names = X.columns

for i, component in enumerate(pca_components):
    print(f"\nPrincipal Component {i+1}:")
    for feature, weight in zip(feature_names, component):
        print(f"{feature}: {weight:.4f}")



# 주성분 구성 시각화 (PC1 ~ PC5이 MFCC 특징으로 어떻게 구성되었는지)

fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
for i in range(5):
    axes[i].barh(feature_names, pca.components_[i], color='blue')
    axes[i].set_title(f'Contribution of Features to PC{i+1}')
    axes[i].set_xlabel('Weight')
plt.tight_layout()
plt.show()
