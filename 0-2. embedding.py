import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. 데이터 로드 (df_combined CSV 파일)
df_combined = pd.read_csv('/content/drive/MyDrive/sound/CSV/combined_data(500).csv')

# 2. Feature와 Label 분리
X = df_combined.drop(columns=['Label', 'Cluster', 'True_Label'])  # 클러스터 및 라벨 정보 제거 (원본 MFCC 벡터만 사용)
y_true = df_combined['True_Label']  # 실제 라벨 (True_Label)


def scale_data(X, method="standard"):
    """
    데이터 스케일링 함수

    Parameters:
    - X: 입력 데이터
    - method: 스케일링 방법 선택 ('standard', 'normalizer', 'robust', 'minmax', 'maxabs')

    Returns:
    - 스케일링된 데이터
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "normalizer":
        scaler = Normalizer()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "maxabs":
        scaler = MaxAbsScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")

    X_scaled = scaler.fit_transform(X)
    return X_scaled

# method: 스케일링 방법 선택 ('standard', 'normalizer', 'robust', 'minmax', 'maxabs')
scaling_method = 'standard'  # 사용자가 원하는 스케일링 방법을 선택
X_scaled = scale_data(X, method=scaling_method)

# 4-1. UMAP 임베딩 적용
def umap_embedding(X_scaled):
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X_scaled)
    return X_umap

# 4-2. t-NSE 임베딩 적용
def tsne_embedding(X_scaled):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    return X_tsne

# 4-3. PCA 임베딩 적용
def pca_embedding(X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def autoencoder_embedding(X_scaled, encoding_dim=2, epochs=50, batch_size=32, learning_rate=0.001):
    input_dim = X_scaled.shape[1]

    # AutoEncoder 모델 정의
    input_layer = Input(shape=(input_dim,))

    # Encoder Layers
    encoded = Dense(128)(input_layer)
    encoded = LeakyReLU(alpha=0.2)(encoded)
    encoded = BatchNormalization()(encoded)

    encoded = Dense(64)(encoded)
    encoded = LeakyReLU(alpha=0.2)(encoded)
    encoded = BatchNormalization()(encoded)

    encoded = Dense(encoding_dim, activation='linear')(encoded)  # Encoding layer

    # Decoder Layers
    decoded = Dense(64)(encoded)
    decoded = LeakyReLU(alpha=0.2)(decoded)
    decoded = BatchNormalization()(decoded)

    decoded = Dense(128)(decoded)
    decoded = LeakyReLU(alpha=0.2)(decoded)
    decoded = BatchNormalization()(decoded)

    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # AutoEncoder 모델
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # 학습
    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    # Encoder 모델로 임베딩 추출
    encoder = Model(input_layer, encoded)
    X_autoencoded = encoder.predict(X_scaled)

    return X_autoencoded

def perform_clustering(X_embedded, y_true, method_name="Embedding"):
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_embedded)
    y_pred = kmeans.labels_

    # 평가
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = max(conf_matrix[0, 0] + conf_matrix[1, 1], conf_matrix[0, 1] + conf_matrix[1, 0]) / len(y_true)

    print(f"{method_name} - Confusion Matrix: \n", conf_matrix)
    print(f"{method_name} - Clustering Accuracy: {accuracy * 100:.2f}%")

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50)
    plt.title(f'{method_name} - K-means Clustering')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    # 실제 라벨 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='coolwarm', edgecolor='k', s=50)
    plt.title(f'{method_name} - True Labels')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    plt.colorbar(label='True Label')
    plt.show()

X_umap = umap_embedding(X_scaled)
perform_clustering(X_umap, y_true, method_name=f"UMAP with {scaling_method}")

X_tsne = tsne_embedding(X_scaled)
perform_clustering(X_tsne, y_true, method_name=f"t-SNE with {scaling_method}")

X_pca = pca_embedding(X_scaled)
perform_clustering(X_pca, y_true, method_name=f"PCA with {scaling_method}")

X_auto = autoencoder_embedding(X_scaled)
perform_clustering(X_auto, y_true, method_name=f"Auto Encoder with {scaling_method}")

# 1. 로지스틱 회귀를 사용한 분류 함수
def logistic_regression_classification(X, y, test_size=0.3, random_state=42):
    """
    로지스틱 회귀를 사용한 분류 함수

    Parameters:
    - X: 임베딩된 피처 벡터
    - y: 예측할 실제 라벨 (True_Label)
    - test_size: 학습/테스트 데이터 분할 비율
    - random_state: 랜덤 시드

    Returns:
    - accuracy: 정확도
    - classification_report: 분류 보고서
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 로지스틱 회귀 모델 학습
    logreg = LogisticRegression(solver='liblinear')  # 작은 데이터셋에 적합한 solver
    logreg.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = logreg.predict(X_test)

    # 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"로지스틱 회귀 정확도: {accuracy * 100:.2f}%")
    print("분류 보고서:\n", class_report)

    return accuracy, class_report

X = X_umap
y = df_combined['True_Label']  # 예측 대상 라벨

# 로지스틱 회귀 사용
logistic_regression_classification(X, y)

X = X_tsne
y = df_combined['True_Label']  # 예측 대상 라벨

# 로지스틱 회귀 사용
logistic_regression_classification(X, y)

X = X_pca
y = df_combined['True_Label']  # 예측 대상 라벨

# 로지스틱 회귀 사용
logistic_regression_classification(X, y)

X = X_auto
y = df_combined['True_Label']  # 예측 대상 라벨

# 로지스틱 회귀 사용
logistic_regression_classification(X, y)

# 1. SVM을 사용한 분류 함수
def svm_classification(X, y, test_size=0.3, random_state=42, kernel='rbf'):
    """
    SVM을 사용한 분류 함수

    Parameters:
    - X: 임베딩된 피처 벡터
    - y: 예측할 실제 라벨 (True_Label)
    - test_size: 학습/테스트 데이터 분할 비율
    - random_state: 랜덤 시드
    - kernel: SVM 커널 ('linear', 'poly', 'rbf', 'sigmoid')

    Returns:
    - accuracy: 정확도
    - classification_report: 분류 보고서
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # SVM 모델 학습 (기본 커널: 'rbf')
    svm_model = SVC(kernel=kernel, random_state=random_state)
    svm_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = svm_model.predict(X_test)

    # 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"SVM ({kernel} 커널) 정확도: {accuracy * 100:.2f}%")
    print("분류 보고서:\n", class_report)

    return accuracy, class_report

X = X_umap
y = df_combined['True_Label']  # 예측 대상 라벨

# SVM 사용 (기본 커널: rbf)
svm_classification(X, y)

# SVM 사용 (선형 커널)
svm_classification(X, y, kernel='sigmoid')

X = X_tsne
y = df_combined['True_Label']  # 예측 대상 라벨

# SVM 사용 (기본 커널: rbf)
svm_classification(X, y)

# SVM 사용 (선형 커널)
svm_classification(X, y, kernel='linear')

X = X_pca
y = df_combined['True_Label']  # 예측 대상 라벨

# SVM 사용 (기본 커널: rbf)
svm_classification(X, y)

# SVM 사용 (선형 커널)
svm_classification(X, y, kernel='linear')

X = X_auto
y = df_combined['True_Label']  # 예측 대상 라벨

# SVM 사용 (기본 커널: rbf)
svm_classification(X, y)

# SVM 사용 (선형 커널)
svm_classification(X, y, kernel='linear')


# 1. XGBoost를 사용한 분류 함수
def xgboost_classification(X, y, test_size=0.3, random_state=42):
    """
    XGBoost를 사용한 분류 함수

    Parameters:
    - X: 임베딩된 피처 벡터
    - y: 예측할 실제 라벨 (True_Label)
    - test_size: 학습/테스트 데이터 분할 비율
    - random_state: 랜덤 시드

    Returns:
    - accuracy: 정확도
    - classification_report: 분류 보고서
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # XGBoost 모델 초기화 및 학습
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = model.predict(X_test)

    # 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"XGBoost 정확도: {accuracy * 100:.2f}%")
    print("분류 보고서:\n", class_report)

    return accuracy, class_report

X = X_umap
y = df_combined['True_Label']  # 예측 대상 라벨

# SVM 사용 (기본 커널: rbf)
xgboost_classification(X, y)


X = X_tsne
y = df_combined['True_Label']  # 예측 대상 라벨

# SVM 사용 (기본 커널: rbf)
xgboost_classification(X, y)

X = X_pca
y = df_combined['True_Label']  # 예측 대상 라벨

# SVM 사용 (기본 커널: rbf)
xgboost_classification(X, y)

X = X_auto
y = df_combined['True_Label']  # 예측 대상 라벨

# SVM 사용 (기본 커널: rbf)
xgboost_classification(X, y)

