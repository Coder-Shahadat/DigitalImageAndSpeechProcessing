import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Load an audio file (replace 'audio_file.wav' with your file)
audio_file = "sound.wav"

# Recognize speech from the audio file
with sr.AudioFile(audio_file) as source:
    audio = recognizer.record(source)

    try:
        recognized_text = recognizer.recognize_google(audio)
        print("Recognized Text:", recognized_text)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Web Speech API; {0}".format(e))
