from gtts import gTTS
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

def text_to_speech(text, output_file):
    tts = gTTS(text)
    tts.save(output_file)

def record_audio(output_file, duration=5, sample_rate=44100):
    print("Recording...")

    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()

    print("Recording finished.")
    write(output_file, sample_rate, recording)

text = "I am shahadat.This is text to speech converter"
output_file_tts = "output_tts.mp3"

text_to_speech(text, output_file_tts)
print("Text-to-speech conversion complete. Saved as", output_file_tts)

output_file_recording = "recorded_audio.wav"
recording_duration = 5  # Duration of the recording in seconds

record_audio(output_file_recording, recording_duration)
print("Audio recording complete. Saved as", output_file_recording)