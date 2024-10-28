import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 꿀벌과 말벌 데이터셋을 불러옵니다
df_mfcc_bee = pd.read_csv('/content/drive/MyDrive/sound/CSV/MFCC_CSV/mfcc_honeybee.csv')
df_mfcc_hornet = pd.read_csv('/content/drive/MyDrive/sound/CSV/MFCC_CSV/mfcc_hornet.csv')

# 정확도 기록을 위한 리스트 초기화
n_samples_list = list(range(100, 6100, 100))
train_accuracies = []
val_accuracies = []
test_accuracies = []
pseudo_test_accuracies = []

# 신뢰도 기준 설정
confidence_threshold = 0.95  # 신뢰도가 이 값을 넘는 경우에만 pseudo-label로 학습

# n_samples의 수를 100부터 6000까지 증가시키며 반복 수행
for n_samples in n_samples_list:
    # 꿀벌 및 말벌 데이터에서 각각 n_samples 만큼 샘플링
    df_mfcc_bee_sampled = df_mfcc_bee.sample(n=n_samples, random_state=42)
    df_mfcc_hornet_sampled = df_mfcc_hornet.sample(n=n_samples, random_state=42)

    # 라벨 추가
    df_mfcc_bee_sampled['Label'] = 'B'
    df_mfcc_hornet_sampled['Label'] = 'H'

    # 데이터 결합
    df_combined = pd.concat([df_mfcc_bee_sampled, df_mfcc_hornet_sampled], axis=0)

    # Train, Test, Validation Set 나누기
    X = df_combined.drop(columns=['Label'])
    y = df_combined['Label']
    y_encoded = pd.get_dummies(y).values.argmax(axis=1)  # Label을 수치형으로 변환

    # Train (60%), Validation (20%), Test (20%) 세트로 분할
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 로지스틱 회귀 모델 학습
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(X_train, y_train)

    # 훈련 데이터에 대한 정확도 계산
    y_train_pred = logreg.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)

    # 검증 데이터에 대한 예측
    y_val_pred = logreg.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_accuracies.append(val_accuracy)  # 검증 데이터에 대한 정확도 기록

    # 테스트 데이터 예측
    y_test_pred = logreg.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)

    # 지도 학습에서 쓴 것 제외한 나머지 데이터 (unlabeled_hornet 및 unlabeled_bee)
    unlabeled_hornet = df_mfcc_hornet.drop(df_mfcc_hornet_sampled.index)
    unlabeled_bee = df_mfcc_bee.drop(df_mfcc_bee_sampled.index)

    # 말벌 및 꿀벌의 unlabeled 데이터를 결합
    unlabeled_data = pd.concat([unlabeled_hornet, unlabeled_bee], axis=0)

    # 예측된 레이블을 기준으로 신뢰도 높은 데이터 선택
    X_unlabeled = unlabeled_data  # 'Label' 열을 제거하지 않음
    if len(X_unlabeled) > 0:
        probs = logreg.predict_proba(X_unlabeled)
        high_confidence_mask = np.max(probs, axis=1) > confidence_threshold

        # 신뢰도 높은 데이터가 없는 경우를 처리
        if np.any(high_confidence_mask):
            # 신뢰도 높은 데이터만 pseudo-labeled로 추가
            pseudo_labeled_data = unlabeled_data[high_confidence_mask].copy()
            y_unlabeled_pred = logreg.predict(X_unlabeled)
            pseudo_labeled_data['Label'] = y_unlabeled_pred[high_confidence_mask]

            # Pseudo-labeled 데이터와 기존 데이터를 결합하여 다시 학습
            df_combined = pd.concat([df_combined, pseudo_labeled_data], ignore_index=True)

            # 다시 학습을 위한 데이터 준비
            X = df_combined.drop(columns=['Label'])
            y = df_combined['Label']
            y_encoded = pd.get_dummies(y).values.argmax(axis=1)

            # Train Set과 Validation Set 재구성
            X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # 로지스틱 회귀 모델 재학습
            logreg = LogisticRegression(solver='liblinear')
            logreg.fit(X_train, y_train)

            # 테스트 데이터에 대한 예측
            y_test_pred = logreg.predict(X_test)
            pseudo_test_accuracy = accuracy_score(y_test, y_test_pred)
            pseudo_test_accuracies.append(pseudo_test_accuracy)
            # print(f"Test Set Accuracy with Pseudo Labeling (n_samples={n_samples}): {pseudo_test_accuracy * 100:.2f}%")
        else:
            print(f"No high confidence pseudo-labeled data for n_samples={n_samples}")
            pseudo_test_accuracies.append(test_accuracy)  # 신뢰도 높은 데이터가 없을 경우 기존 테스트 정확도 사용
    else:
        print(f"No unlabeled data for n_samples={n_samples}")
        pseudo_test_accuracies.append(test_accuracy)  # unlabeled 데이터가 없을 경우 기존 테스트 정확도 사용

# 정확도 변화 시각화
plt.figure(figsize=(10, 6))
plt.plot(n_samples_list, train_accuracies, marker='s', linestyle='-', color='g', label='Train Accuracy')
plt.plot(n_samples_list, val_accuracies, marker='x', linestyle='--', color='r', label='Validation Accuracy')
plt.plot(n_samples_list, test_accuracies, marker='o', linestyle='-', color='b', label='Test Accuracy')
plt.plot(n_samples_list, pseudo_test_accuracies, marker='*', linestyle='-', color='y', label='Pseudo Test Accuracy')

plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.title('Train, Test, and Validation Accuracy vs. Number of Samples (Confidence Threshold: 0.95)')
plt.grid(True)
plt.legend()
plt.show()
