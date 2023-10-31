import librosa
import numpy as np
import matplotlib.pyplot as plt

# Create an array to represent our audio data
sr = 22050  # Sample rate
T = 5.0    # Total duration in seconds
t = np.linspace(0, T, int(T*sr), endpoint=False) # Time array

# Generate a signal with two sine waves and some silence in between
x = 0.5 * np.sin(2 * np.pi * 220 * t)            # Sine wave at 220 Hz
x[int(T//2*sr):int((T//2+0.5)*sr)] = 0           # Silence in the middle
x += 0.5 * np.sin(2 * np.pi * 224 * t)           # Sine wave at 224 Hz

# Let's add some noise
x += 0.01 * np.random.normal(size=len(t))

window_length = 500  # Length of the moving average window
threshold = 0.4  # Amplitude level below which sound is considered 'quiet'

# Take the moving average of the absolute values of the signal
abs_x = np.abs(x)
moving_avg = np.convolve(abs_x, np.ones(window_length)/window_length, mode='valid')

# Find where the moving average is below the threshold
quiet_sections = np.where(moving_avg < threshold)[0] + window_length // 2

# Determine how many samples represent 1/4 second
quarter_second_samples = int(0.25 * sr)

# Group quiet sections into consecutive sequences
def group_consecutive(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0]+1)

groups = group_consecutive(quiet_sections)

# Filter out groups that are shorter than 1/4 second
long_silences = [g for g in groups if len(g) >= quarter_second_samples]

# Find the start and end of these long silence sections
long_silences_ranges = [(g[0], g[-1]) for g in long_silences]

# Plotting original signal with long quiet sections
plt.figure(figsize=(20, 4))
plt.plot(t, x, label="Original Signal")
for start, end in long_silences_ranges:
    plt.axvspan(t[start], t[end], color='red', alpha=0.2, label="Long Quiet Section")
plt.legend()
plt.show()
