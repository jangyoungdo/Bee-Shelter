# Import required libraries
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
import librosa
import pandas as pd

# Google Drive mounting (only if using Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Define utility functions
def butter_lowpass(cutoff, fs, order=5):
    """
    저역 통과 필터 설계 (Butterworth 필터).
    
    Parameters:
    - cutoff: 차단 주파수 (Hz)
    - fs: 샘플링 주파수 (Hz)
    - order: 필터의 차수 (default=5)
    
    Returns:
    - b, a: 필터 계수
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5, padlen=0):
    """
    입력 신호에 저역 통과 필터 적용.
    
    Parameters:
    - data: 입력 신호 데이터 (numpy 배열)
    - cutoff: 차단 주파수 (Hz)
    - fs: 샘플링 주파수 (Hz)
    - order: 필터의 차수 (default=5)
    - padlen: 필터 패딩 길이 (default=0)
    
    Returns:
    - filtered_data: 필터링된 데이터
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    filtered_data = filtfilt(b, a, data, padlen=padlen)
    return filtered_data

def load_and_concatenate_wavs(file_list):
    """
    여러 개의 WAV 파일을 불러와서 하나로 결합.
    
    Parameters:
    - file_list: 결합할 파일 경로 리스트
    
    Returns:
    - concatenated_data: 결합된 오디오 데이터 (numpy 배열)
    - fs: 샘플링 주파수 (모든 파일에서 동일하다고 가정)
    """
    concatenated_data = []
    fs = None

    for file in file_list:
        try:
            data, sample_rate = librosa.load(file, sr=None)  # 원본 샘플링 레이트 유지
            if fs is None:
                fs = sample_rate
            elif fs != sample_rate:
                raise ValueError("모든 파일의 샘플링 레이트가 같아야 합니다.")
            concatenated_data.append(data)
        except Exception as e:
            print(f"파일 {file}을(를) 불러오는 중 에러 발생: {e}")

    concatenated_data = np.concatenate(concatenated_data)  # 모든 파일 결합
    return concatenated_data, fs

def process_and_save_wav(input_data, fs, output_path, cutoff_frequency):
    """
    결합된 WAV 데이터를 필터링하고 저장.
    
    Parameters:
    - input_data: 입력 오디오 데이터 (numpy 배열)
    - fs: 샘플링 주파수
    - output_path: 저장할 파일 경로
    - cutoff_frequency: 저역 통과 필터의 차단 주파수 (Hz)
    
    Returns:
    - filtered_data: 필터링된 데이터
    """
    filtered_data = apply_lowpass_filter(input_data, cutoff_frequency, fs)
    try:
        wavfile.write(output_path, fs, np.int16(filtered_data * 32767))  # 16비트 PCM 포맷으로 저장
        print(f"필터링된 오디오 파일이 {output_path}에 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 에러 발생: {e}")
    return filtered_data

def extract_mfcc_features(filtered_data, fs):
    """
    필터링된 데이터에서 전체 MFCC를 추출하여 특징 벡터로 반환.
    
    Parameters:
    - filtered_data: 필터링된 오디오 데이터 (numpy 배열)
    - fs: 샘플링 주파수
    
    Returns:
    - mfcc_mean: 추출된 MFCC 특징 벡터 (평균값)
    """
    # MFCC 추출 (fmax=500으로 제한)
    mfcc = librosa.feature.mfcc(y=filtered_data, sr=fs, n_mfcc=13, n_mels=128, fmax=500)
    mfcc_mean = np.mean(mfcc, axis=1)  # MFCC 평균값 추출
    return mfcc_mean

# 파일 경로 설정
honeybee_file = '/content/drive/MyDrive/sound/produced/Noise_cancel/Honeybee/nn_REC001.WAV.wav'
output_honeybee = '/content/drive/MyDrive/sound/produced/Noise_cancel/produced/Honeybee_500/honeybee_filtered.wav'
hornet_files = [
    '/content/drive/MyDrive/sound/produced/Noise_cancel/Hornet/nn_REC002.WAV.wav',
    '/content/drive/MyDrive/sound/produced/Noise_cancel/Hornet/nn_REC003.WAV.wav',
    '/content/drive/MyDrive/sound/produced/Noise_cancel/Hornet/nn_REC004.WAV.wav',
    '/content/drive/MyDrive/sound/produced/Noise_cancel/Hornet/nn_REC005.WAV.wav',
    '/content/drive/MyDrive/sound/produced/Noise_cancel/Hornet/nn_REC006.WAV.wav',
    '/content/drive/MyDrive/sound/produced/Noise_cancel/Hornet/nn_REC007.WAV.wav',
    '/content/drive/MyDrive/sound/produced/Noise_cancel/Hornet/nn_REC008.WAV.wav',
    '/content/drive/MyDrive/sound/produced/Noise_cancel/Hornet/nn_REC009.WAV.wav'
]
output_hornet = '/content/drive/MyDrive/sound/produced/Noise_cancel/produced/Hornet_500/hornet_filtered.wav'

# 저역 통과 필터 주파수 설정
cutoff_frequency = 500  # Hz

# Honeybee 파일 처리
filtered_bee = process_and_save_wav(*load_and_concatenate_wavs([honeybee_file]), output_honeybee, cutoff_frequency)

# Hornet 파일 처리 (여러 파일 결합 후 필터링)
concatenated_hornet_data, fs_hornet = load_and_concatenate_wavs(hornet_files)
filtered_hornet = process_and_save_wav(concatenated_hornet_data, fs_hornet, output_hornet, cutoff_frequency)

# 오디오 파일 다운샘플링 (필터링된 데이터)
fs_target = 2000  # 2kHz 샘플링 주파수 설정
filtered_data_bee_resampled = librosa.resample(filtered_bee, orig_sr=fs_hornet, target_sr=fs_target)
filtered_data_hornet_resampled = librosa.resample(filtered_hornet, orig_sr=fs_hornet, target_sr=fs_target)

# MFCC 특징 추출
honeybee_features = extract_mfcc_features(filtered_data_bee_resampled, fs_target)
hornet_features = extract_mfcc_features(filtered_data_hornet_resampled, fs_target)

# 특징 데이터를 pandas DataFrame으로 변환
honeybee_df = pd.DataFrame([honeybee_features], columns=[f'MFCC_{i+1}' for i in range(len(honeybee_features))])
hornet_df = pd.DataFrame([hornet_features], columns=[f'MFCC_{i+1}' for i in range(len(hornet_features))])

# 클래스 레이블 추가 (0: Honeybee, 1: Hornet)
honeybee_df['label'] = 0
hornet_df['label'] = 1

# 두 데이터를 결합하여 CSV 파일로 저장
features_df = pd.concat([honeybee_df, hornet_df], ignore_index=True)
csv_path = '/content/drive/MyDrive/sound/produced/Noise_cancel/produced/mfcc_features.csv'
features_df.to_csv(csv_path, index=False)
print(f"MFCC 특징값이 {csv_path}에 저장되었습니다.")