import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Simulation parameters
fs = 100  # Sampling frequency (Hz)
duration = 10  # seconds
t = np.linspace(0, duration, fs * duration)  # Time vector

# Simulating different earthquake wave components
p_wave = np.sin(2 * np.pi * 3 * t)  # Primary (P) wave - Low frequency
s_wave = 0.5 * np.sin(2 * np.pi * 10 * t)  # Secondary (S) wave - Medium frequency
surface_wave = 0.3 * np.sin(2 * np.pi * 20 * t)  # Surface wave - High frequency

# Simulating an earthquake shockwave (sudden energy release at t=2s)
shockwave = np.exp(-((t - 2) ** 2) / 0.1)  # Gaussian spike

# Combine all waves and add shockwave effect
earthquake_wave = (p_wave + s_wave + surface_wave) * shockwave

# Add random noise to mimic real seismic activity
noise = np.random.normal(0, 0.1, earthquake_wave.shape)
earthquake_wave += noise

# Function to apply Butterworth low-pass filter (for noise reduction)
def butter_lowpass_filter(data, cutoff=15, fs=100, order=5):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

# Apply filtering to remove high-frequency noise
filtered_wave = butter_lowpass_filter(earthquake_wave)

# FFT Analysis
fft_values = np.fft.fft(earthquake_wave)  # Compute FFT
frequencies = np.fft.fftfreq(len(t), d=1/fs)  # Compute frequency bins

# Plot simulated earthquake wave
plt.figure(figsize=(12, 5))
plt.plot(t, earthquake_wave, label="Simulated Earthquake Wave", color='b')
plt.plot(t, filtered_wave, label="Filtered Wave", color='r', linestyle="dashed")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Simulated Earthquake Wave (Time Domain)")
plt.legend()
plt.grid()
plt.show()

# Plot FFT spectrum (Frequency Analysis)
plt.figure(figsize=(12, 5))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_values)[:len(frequencies)//2], color='g')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Spectrum of Earthquake Wave")
plt.grid()
plt.show()
