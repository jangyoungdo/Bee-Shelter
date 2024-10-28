def butter_lowpass(cutoff, fs, order=5):
    """저역 통과 필터 생성 함수"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    """저역 통과 필터 적용 함수"""
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, data)

def preprocess_audio(audio_path, cutoff=500, target_sr=2000):
    """오디오 파일 전처리 함수: 저역 필터링 및 샘플링 주파수 변경"""
    y, sr = librosa.load(audio_path, sr=None)  # 원본 샘플링 레이트 유지
    filtered_data = lowpass_filter(y, cutoff, sr)
    resampled_data = librosa.resample(filtered_data, orig_sr=sr, target_sr=target_sr)
    return resampled_data, target_sr

def extract_mfcc_features(data, fs, n_mfcc=13):
    """MFCC 특징 추출 함수"""
    mfcc = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=n_mfcc, n_mels=128, fmax=500)
    return np.mean(mfcc, axis=1)

def process_and_save_features(audio_files, cutoff=500, target_sr=2000, output_csv="mfcc_features.csv"):
    """오디오 파일 목록을 전처리하고, 특징을 추출하여 CSV 파일로 저장"""
    features = []
    labels = []

    for file_path, label in audio_files:
        processed_data, fs = preprocess_audio(file_path, cutoff=cutoff, target_sr=target_sr)
        mfcc_features = extract_mfcc_features(processed_data, fs)
        features.append(mfcc_features)
        labels.append(label)
    
