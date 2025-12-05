import pyaudio
import numpy as np
import time
from collections import deque

# 🎛 Audio settings
BUFFER_SIZE = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100  # Standard sample rate

# 🎹 Notes in solfège notation
SOLFEGE_NOTES = [
    "La0", "La#0", "Si0", "Do1", "Do#1", "Re1", "Re#1", "Mi1", "Fa1", "Fa#1", "Sol1", "Sol#1",
    "La1", "La#1", "Si1", "Do2", "Do#2", "Re2", "Re#2", "Mi2", "Fa2", "Fa#2", "Sol2", "Sol#2",
    "La2", "La#2", "Si2", "Do3", "Do#3", "Re3", "Re#3", "Mi3", "Fa3", "Fa#3", "Sol3", "Sol#3",
    "La3", "La#3", "Si3", "Do4", "Do#4", "Re4", "Re#4", "Mi4", "Fa4", "Fa#4", "Sol4", "Sol#4",
    "La4", "La#4", "Si4", "Do5", "Do#5", "Re5", "Re#5", "Mi5", "Fa5", "Fa#5", "Sol5", "Sol#5",
    "La5", "La#5", "Si5", "Do6", "Do#6", "Re6", "Re#6", "Mi6", "Fa6", "Fa#6", "Sol6", "Sol#6",
    "La6", "La#6", "Si6", "Do7", "Do#7", "Re7", "Re#7", "Mi7", "Fa7", "Fa#7", "Sol7", "Sol#7",
    "La7", "La#7", "Si7", "Do8"
]

# 🎼 Convert frequency to note
def frequency_to_note(freq):
    if freq == 0:
        return None
    la4 = 440.0  # Reference for A4
    note_number = 12 * np.log2(freq / la4) + 49
    note_index = round(note_number) - 1
    if 0 <= note_index < len(SOLFEGE_NOTES):
        return SOLFEGE_NOTES[note_index]
    return None

# 🎤 Start PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=BUFFER_SIZE)

print("🎶 Start playing the piano... (Press Ctrl+C to stop)")

prev_note = None
note_history = deque(maxlen=3)  # Stores last 3 detected notes
last_time = time.time()

# 🎚 Noise threshold: Ignore weak signals (we increase this to filter background noise)
noise_threshold = 10  # This is adjustable

# 🎶 Silence time threshold: minimum time to wait before recognizing a new note
silence_threshold = 0.5  # Delay to prevent picking up extra notes after releasing

try:
    while True:
        # 🎙 Read audio data
        audio_data = np.frombuffer(stream.read(BUFFER_SIZE, exception_on_overflow=False), dtype=np.float32)

        # 🎛 Apply FFT
        fft_result = np.abs(np.fft.rfft(audio_data))
        frequencies = np.fft.rfftfreq(len(audio_data), 1.0 / RATE)

        # 🎼 Find the strongest frequency (dominant note)
        max_index = np.argmax(fft_result)
        max_freq = frequencies[max_index]

        # 🎵 Convert to musical note
        detected_note = frequency_to_note(max_freq)

        # 🎚 Noise filtering: Ignore weak signals when no piano sound is detected
        if detected_note and fft_result[max_index] > noise_threshold:
            note_history.append(detected_note)

            # 🎶 Only print if the note is stable (the same for last 3 detections)
            if len(note_history) == 3 and all(n == detected_note for n in note_history):
                # Print only if the note is different or time passed since last print
                if detected_note != prev_note or time.time() - last_time > silence_threshold:
                    print(f"🎵 {detected_note}")
                    prev_note = detected_note
                    last_time = time.time()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
