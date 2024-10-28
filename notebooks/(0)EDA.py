#!/usr/bin/env python
# coding: utf-8

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydub import AudioSegment
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and adjust audio files
def load_and_adjust_audio(file1, file2, output1, output2):
    audio1 = AudioSegment.from_file(file1)
    audio2 = AudioSegment.from_file(file2)
    length = min(len(audio1), len(audio2))
    audio1, audio2 = audio1[:length], audio2[:length]
    audio1.export(output1, format='wav')
    audio2.export(output2, format='wav')
    print("Adjusted both audio files to the same length.")

load_and_adjust_audio('/content/drive/MyDrive/sound/original/honeybee/internal/bee_2.WAV',
                      '/content/drive/MyDrive/sound/original/hornet/REC002.WAV',
                      '/content/drive/MyDrive/sound/produced/same_len/honeybee/honeybee.wav',
                      '/content/drive/MyDrive/sound/produced/same_len/hornet/hornet.wav')

# Load audio data
sr = 44100
honeybee_path = '/content/drive/MyDrive/sound/produced/same_len/honeybee/honeybee.wav'
hornet_path = '/content/drive/MyDrive/sound/produced/same_len/hornet/hornet.wav'
y_bee, _ = librosa.load(honeybee_path, sr=sr)
y_hornet, _ = librosa.load(hornet_path, sr=sr)

# Downsample to 2 kHz
fs_target = 2000
y_bee_resampled = librosa.resample(y_bee, orig_sr=sr, target_sr=fs_target)
y_hornet_resampled = librosa.resample(y_hornet, orig_sr=sr, target_sr=fs_target)

# Extract MFCC features
def extract_mfcc(audio, fs_target, n_mfcc=13, n_mels=128):
    return librosa.feature.mfcc(y=audio, sr=fs_target, n_mfcc=n_mfcc, n_mels=n_mels)

mfcc_bee = extract_mfcc(y_bee_resampled, fs_target)
mfcc_hornet = extract_mfcc(y_hornet_resampled, fs_target)

# Convert MFCCs to DataFrame
def mfcc_to_dataframe(mfcc, label):
    df = pd.DataFrame(mfcc.T, columns=[f'MFCC_{i+1}' for i in range(mfcc.shape[0])])
    df['Label'] = label
    return df

df_mfcc_bee = mfcc_to_dataframe(mfcc_bee, 'B')
df_mfcc_hornet = mfcc_to_dataframe(mfcc_hornet, 'H')

# Combine data
df_combined = pd.concat([df_mfcc_bee, df_mfcc_hornet], axis=0)

# Data Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_combined.drop(columns=['Label']))

# K-means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(df_scaled)
df_combined['Cluster'] = kmeans.labels_

# Encode True Labels
label_encoder = LabelEncoder()
df_combined['True_Label'] = label_encoder.fit_transform(df_combined['Label'])

# Save combined data to CSV
df_combined.to_csv('/content/drive/MyDrive/sound/CSV/combined_data.csv', index=False)

# Calculate Clustering Accuracy
conf_matrix = confusion_matrix(df_combined['True_Label'], df_combined['Cluster'])
accuracy = max(conf_matrix[0, 0] + conf_matrix[1, 1], conf_matrix[0, 1] + conf_matrix[1, 0]) / len(df_combined)
print(f"Clustering Accuracy: {accuracy * 100:.2f}%")

# Visualize Clustering Results
def visualize_pca(df_scaled, labels, title, label_desc):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label=label_desc)
    plt.show()

visualize_pca(df_scaled, df_combined['Cluster'], 'K-means Clustering of MFCCs', 'Cluster')
visualize_pca(df_scaled, df_combined['True_Label'], 'True Labels of MFCCs', 'True Label')

# Logistic Regression
X = df_combined.drop(columns=['Label', 'Cluster', 'True_Label'])
y = df_combined['True_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Visualization functions for waveform, spectrogram, and audio features
def plot_waveform(y, sr, title):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.show()

def plot_spectrogram(y, sr, title):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

plot_waveform(y_bee, sr, 'Waveform (HoneyBee)')
plot_waveform(y_hornet, sr, 'Waveform (Hornet)')
plot_spectrogram(y_bee, sr, 'Spectrogram (HoneyBee)')
plot_spectrogram(y_hornet, sr, 'Spectrogram (Hornet)')

# Additional Feature Visualization
def plot_feature(y, feature_func, title):
    feature = feature_func(y)[0]
    plt.figure(figsize=(14, 5))
    plt.plot(feature)
    plt.title(title)
    plt.show()

plot_feature(y_bee, lambda x: librosa.feature.rms(y=x), 'RMS Energy (HoneyBee)')
plot_feature(y_hornet, lambda x: librosa.feature.rms(y=x), 'RMS Energy (Hornet)')
plot_feature(y_bee, lambda x: librosa.feature.zero_crossing_rate(y=x), 'Zero-Crossing Rate (HoneyBee)')
plot_feature(y_hornet, lambda x: librosa.feature.zero_crossing_rate(y=x), 'Zero-Crossing Rate (Hornet)')

