import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load a speech signal (replace 'speech.wav' with your file)
speech_signal, Fs = librosa.load('New_Real_Conversation_Lessons.mp3', sr=None)

# Parameters for spectrogram
window_size = int(0.02 * Fs)  # Window size in samples (20 ms)
overlap = int(0.5 * window_size)  # Overlap between windows (50%)

# Create and plot the spectrogram
plt.figure(figsize=(12, 8))
D = librosa.amplitude_to_db(np.abs(librosa.stft(speech_signal, n_fft=window_size, hop_length=overlap)),
                            ref=np.max)
librosa.display.specshow(D, sr=Fs, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Speech Signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()
