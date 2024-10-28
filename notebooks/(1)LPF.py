# Converted Python script from Jupyter Notebook

# Imports and initial setup
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import librosa
import librosa.display

# Function definitions
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Load audio file
audio_path = 'your_audio_file_here.wav'
y, sr = librosa.load(audio_path, sr=None)

# Apply low-pass filters
cutoff1 = 1000
cutoff2 = 500

filtered_y1 = lowpass_filter(y, cutoff1, sr)
filtered_y2 = lowpass_filter(y, cutoff2, sr)

# Plot original and filtered signals
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Original Signal')

plt.subplot(3, 1, 2)
librosa.display.waveshow(filtered_y1, sr=sr)
plt.title(f'Low-pass Filtered Signal (cutoff={cutoff1} Hz)')

plt.subplot(3, 1, 3)
librosa.display.waveshow(filtered_y2, sr=sr)
plt.title(f'Low-pass Filtered Signal (cutoff={cutoff2} Hz)')

plt.tight_layout()
plt.show()
