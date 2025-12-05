import pyaudio
import numpy as np
import time
from collections import deque

# Config
BUFFER_SIZE = 4096           # bigger -> better frequency resolution, more latency
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

# Note names (same list or generate from MIDI mapping)
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
# Helpers
def freq_from_peak_parabolic(fft_mag, freq_bins):
    # find peak bin
    i = np.argmax(fft_mag)
    if i == 0 or i == len(fft_mag)-1:
        return freq_bins[i]
    # parabolic interpolation for peak location
    alpha = fft_mag[i-1]
    beta  = fft_mag[i]
    gamma = fft_mag[i+1]
    p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)  # offset from bin i
    peak_bin = i + p
    return peak_bin * (freq_bins[1] - freq_bins[0])

def frequency_to_note_midi(freq):
    if freq <= 0:
        return None
    # MIDI note number formula: 69 -> A4 = 440 Hz
    midi = 69 + 12 * np.log2(freq / 440.0)
    midi_rounded = int(round(midi))
    note_index = midi_rounded - 21  # if SOLFEGE_NOTES[0] == A0 (MIDI 21)
    if 0 <= note_index < len(SOLFEGE_NOTES):
        return SOLFEGE_NOTES[note_index]
    return None

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=BUFFER_SIZE)

note_history = deque(maxlen=3)
prev_note = None
last_time = time.time()

# dB threshold relative to RMS
MIN_DB = -40.0  # signals below -40 dB (relative) ignored

try:
    while True:
        data = np.frombuffer(stream.read(BUFFER_SIZE, exception_on_overflow=False), dtype=np.float32)
        # compute RMS in dB for thresholding
        rms = np.sqrt(np.mean(data**2) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)

        # apply a Hann window to reduce leakage
        windowed = data * np.hanning(len(data))

        # zero-pad for finer bin spacing (optional)
        ZP = 4 * len(windowed)
        fft_mag = np.abs(np.fft.rfft(windowed, n=ZP))
        freqs = np.fft.rfftfreq(ZP, 1.0 / RATE)

        # find refined peak frequency
        peak_freq = freq_from_peak_parabolic(fft_mag, freqs)

        # only proceed if signal is loud enough
        if db > MIN_DB:
            note = frequency_to_note_midi(peak_freq)
            if note:
                note_history.append(note)
                if len(note_history) == note_history.maxlen and all(n == note for n in note_history):
                    if note != prev_note or (time.time() - last_time) > 0.4:
                        print(f"🎵 {note}  (freq {peak_freq:.1f} Hz, db {db:.1f})")
                        prev_note = note
                        last_time = time.time()
        else:
            # optionally clear history when silence
            note_history.clear()

except KeyboardInterrupt:
    print("Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
