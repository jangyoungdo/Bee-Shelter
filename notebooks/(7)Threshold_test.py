import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 정확도 기록을 위한 딕셔너리 초기화
n_samples_list = list(range(100, 6100, 100))
train_accuracies = []
val_accuracies = []
test_accuracies = []
pseudo_test_accuracies_by_threshold = {threshold: [] for threshold in np.arange(0.55, 1.0, 0.05)}

# n_samples의 수를 100부터 6000까지 증가시키며 반복 수행
for n_samples in n_samples_list:
    # 꿀벌 및 말벌 데이터에서 각각 n_samples 만큼 샘플링
    df_mfcc_bee_sampled = df_mfcc_bee.sample(n=n_samples, random_state=42)
    df_mfcc_hornet_sampled = df_mfcc_hornet.sample(n=n_samples, random_state=42)

    # 라벨 추가 및 데이터 결합
    df_mfcc_bee_sampled['Label'] = 'B'
    df_mfcc_hornet_sampled['Label'] = 'H'
    df_combined = pd.concat([df_mfcc_bee_sampled, df_mfcc_hornet_sampled], axis=0)

    # Train, Test, Validation Set 나누기
    X = df_combined.drop(columns=['Label'])
    y = pd.get_dummies(df_combined['Label']).values.argmax(axis=1)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 로지스틱 회귀 모델 학습 및 정확도 계산
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(X_train, y_train)
    train_accuracies.append(accuracy_score(y_train, logreg.predict(X_train)))
    val_accuracies.append(accuracy_score(y_val, logreg.predict(X_val)))
    test_accuracies.append(accuracy_score(y_test, logreg.predict(X_test)))

    # 지도 학습에서 제외한 나머지 데이터 (unlabeled_hornet 및 unlabeled_bee)
    unlabeled_data = pd.concat([df_mfcc_hornet.drop(df_mfcc_hornet_sampled.index),
                                df_mfcc_bee.drop(df_mfcc_bee_sampled.index)], axis=0)

    # 예측된 레이블을 기준으로 신뢰도 높은 데이터 선택 및 Pseudo Labeling 반복
    X_unlabeled = unlabeled_data
    probs = logreg.predict_proba(X_unlabeled)

    for threshold in pseudo_test_accuracies_by_threshold.keys():
        high_confidence_mask = np.max(probs, axis=1) > threshold
        pseudo_labeled_data = unlabeled_data[high_confidence_mask].copy()
        pseudo_labeled_data['Label'] = logreg.predict(X_unlabeled)[high_confidence_mask]

        # Pseudo-labeled 데이터와 기존 데이터를 결합하여 다시 학습
        df_combined_pseudo = pd.concat([df_combined, pseudo_labeled_data], ignore_index=True)
        X_pseudo = df_combined_pseudo.drop(columns=['Label'])
        y_pseudo = pd.get_dummies(df_combined_pseudo['Label']).values.argmax(axis=1)
        X_train_pseudo, X_temp_pseudo, y_train_pseudo, y_temp_pseudo = train_test_split(
            X_pseudo, y_pseudo, test_size=0.4, random_state=42)
        X_val_pseudo, X_test_pseudo, y_val_pseudo, y_test_pseudo = train_test_split(
            X_temp_pseudo, y_temp_pseudo, test_size=0.5, random_state=42)

        # 로지스틱 회귀 모델 학습 및 테스트 데이터에 대한 정확도 계산
        logreg_pseudo = LogisticRegression(solver='liblinear')
        logreg_pseudo.fit(X_train_pseudo, y_train_pseudo)
        pseudo_test_accuracy = accuracy_score(y_test_pseudo, logreg_pseudo.predict(X_test_pseudo))
        pseudo_test_accuracies_by_threshold[threshold].append(pseudo_test_accuracy)

# 정확도 변화 시각화
plt.figure(figsize=(12, 8))
for threshold, accuracies in pseudo_test_accuracies_by_threshold.items():
    plt.plot(n_samples_list, accuracies, marker='*', linestyle='-', label=f'Threshold: {threshold:.2f}')

plt.xlabel('Number of Samples')
plt.ylabel('Pseudo Test Accuracy')
plt.title('Pseudo Test Accuracy vs. Number of Samples at Different Confidence Thresholds')
plt.grid(True)
plt.legend()
plt.show()
