bee_sample_sizes = list(range(100, 6100, 100))
hornet_sample_sizes = list(range(100, 6100, 100))
pseudo_test_accuracies = []

# 신뢰도 기준 설정
confidence_threshold = 0.95  # 신뢰도가 이 값을 넘는 경우에만 pseudo-label로 학습

# 로지스틱 회귀 모델 초기화 함수
def train_logistic_regression(X_train, y_train):
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(X_train, y_train)
    return logreg

# Train, Validation, Test 세트를 나누는 함수
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# 샘플링 및 데이터 준비 함수
def prepare_data(df_bee, df_hornet, bee_samples, hornet_samples):
    df_bee_sampled = df_bee.sample(n=bee_samples, random_state=42)
    df_hornet_sampled = df_hornet.sample(n=hornet_samples, random_state=42)
    df_bee_sampled['Label'] = 'B'
    df_hornet_sampled['Label'] = 'H'
    return pd.concat([df_bee_sampled, df_hornet_sampled], axis=0)

# 수도 라벨링 및 정확도 계산 함수
def apply_pseudo_labeling(logreg, unlabeled_data, confidence_threshold):
    if len(unlabeled_data) == 0:
        return None, 0
    
    probs = logreg.predict_proba(unlabeled_data)
    high_confidence_mask = np.max(probs, axis=1) > confidence_threshold

    if np.any(high_confidence_mask):
        pseudo_labeled_data = unlabeled_data[high_confidence_mask].copy()
        y_unlabeled_pred = logreg.predict(unlabeled_data)
        pseudo_labeled_data['Label'] = y_unlabeled_pred[high_confidence_mask]
        return pseudo_labeled_data, 1
    return None, 0

# 메인 루프
for bee_samples in bee_sample_sizes:
    for hornet_samples in hornet_sample_sizes:
        # 데이터 준비
        df_combined = prepare_data(df_mfcc_bee, df_mfcc_hornet, bee_samples, hornet_samples)
        X = df_combined.drop(columns=['Label'])
        y = pd.get_dummies(df_combined['Label']).values.argmax(axis=1)

        # Train, Validation, Test 세트 분할
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # 초기 모델 학습
        logreg = train_logistic_regression(X_train, y_train)

        # 지도 학습에서 제외된 나머지 데이터 준비
        unlabeled_hornet = df_mfcc_hornet.drop(df_mfcc_hornet.index.intersection(df_combined.index))
        unlabeled_bee = df_mfcc_bee.drop(df_mfcc_bee.index.intersection(df_combined.index))
        unlabeled_data = pd.concat([unlabeled_hornet, unlabeled_bee], axis=0)

        # 수도 라벨링 적용
        pseudo_labeled_data, pseudo_label_applied = apply_pseudo_labeling(logreg, unlabeled_data, confidence_threshold)

        # 수도 라벨링이 적용된 경우 데이터 결합 후 재학습
        if pseudo_label_applied:
            df_combined = pd.concat([df_combined, pseudo_labeled_data], ignore_index=True)
            X = df_combined.drop(columns=['Label'])
            y = pd.get_dummies(df_combined['Label']).values.argmax(axis=1)
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
            logreg = train_logistic_regression(X_train, y_train)

        # 테스트 데이터에 대한 예측 및 정확도 계산
        y_test_pred = logreg.predict(X_test)
        pseudo_test_accuracy = accuracy_score(y_test, y_test_pred) if pseudo_label_applied else 0

        # 결과 기록 및 출력
        pseudo_test_accuracies.append((bee_samples, hornet_samples, pseudo_test_accuracy))
        print(f"Bee Samples: {bee_samples}, Hornet Samples: {hornet_samples}, Pseudo Test Accuracy: {pseudo_test_accuracy * 100:.2f}%")

# 결과를 DataFrame으로 변환하여 시각화 준비
df_results = pd.DataFrame(pseudo_test_accuracies, columns=['Bee Samples', 'Hornet Samples', 'Pseudo Test Accuracy'])

# 시각화: 꿀벌 샘플 수를 고정하고, 말벌 샘플 수에 따른 정확도 변화를 확인
plt.figure(figsize=(12, 8))
for bee_samples in bee_sample_sizes:
    subset = df_results[df_results['Bee Samples'] == bee_samples]
    plt.plot(subset['Hornet Samples'], subset['Pseudo Test Accuracy'], marker='o', linestyle='-', label=f'Bee Samples: {bee_samples}')

plt.xlabel('Number of Hornet Samples')
plt.ylabel('Pseudo Test Accuracy')
plt.title('Pseudo Test Accuracy vs. Number of Hornet Samples at Different Bee Sample Sizes')
plt.grid(True)
plt.legend()
plt.show()
