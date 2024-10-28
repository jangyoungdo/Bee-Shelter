import librosa
import pandas as pd

# 오디오 파일 로드
y, sr = librosa.load('/content/drive/MyDrive/sound/produced/Noise_cancel/produced/Hornet_500/hornet_filtered.wav', sr=None)

# MFCC 계산
n_mfcc = 13
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

# 가중치가 높은 5개의 독립변수 확인 (이전 코드에서 얻은 df_coefficients 사용)
df_coefficients = pd.DataFrame({  # 예시 데이터프레임 생성
    'Feature': [f'MFCC_{i+1}' for i in range(n_mfcc)],
    'Coefficient': [0.1 * i for i in range(n_mfcc)]
}).sort_values(by='Coefficient', ascending=False)

top_5_features = df_coefficients.head(5)['Feature']

# Mel 필터 뱅크 주파수 대역 계산
n_mels = 128
fmin = 0
fmax = sr // 2
mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

# 각 MFCC 계수가 대표하는 주파수 대역 추정 (범위)
for feature in top_5_features:
    # Feature 이름에서 숫자 부분만 추출 (예: 'MFCC_4'에서 숫자 4 추출)
    feature_index = int(''.join(filter(str.isdigit, feature))) - 1

    # 해당 MFCC가 담당하는 주파수 대역 범위를 찾음
    mel_band_min = mel_frequencies[feature_index]
    mel_band_max = mel_frequencies[feature_index + 1] if feature_index + 1 < len(mel_frequencies) else fmax
    print(f"{feature} corresponds to frequency range {mel_band_min:.2f} Hz - {mel_band_max:.2f} Hz")
