import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks
from scipy.fft import fft, fftfreq, ifft

# ========== PART 1: SENSING using LFM ==========

# Parameters
fs = 1000                    # Sampling frequency (Hz)
T = 10                       # Total observation time (seconds)
N = fs * T                   # Total samples

# LFM Signal Parameters
f0 = 20                      # Start frequency
f1 = 200                     # End frequency

# Time vector
t = np.linspace(0, T, N)

# Generate Dynamic Heart and Breath Rates
# Heart Rate and Breath Rate will oscillate with time
def generate_vital_signs(t):
    heart_rate = 60 + 10 * np.sin(2 * np.pi * 0.05 * t)  # Heart rate oscillates between 60 and 70
    breath_rate = 15 + 2 * np.sin(2 * np.pi * 0.01 * t)   # Breath rate oscillates between 15 and 17
    return heart_rate, breath_rate

# Transmitted LFM signal (chirp)
transmitted_signal = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

# Simulate Chest Movement
heart_rate, breath_rate = generate_vital_signs(t)
breath_motion = 0.01 * np.sin(2 * np.pi * breath_rate * t)
heart_motion = 0.002 * np.sin(2 * np.pi * heart_rate * t)
total_motion = breath_motion + heart_motion

# Received Signal with Phase Modulation (dynamic)
received_signal = transmitted_signal * np.exp(1j * 2 * np.pi * total_motion)

# Extract Phase
received_phase = np.unwrap(np.angle(received_signal))

# Remove trend
phase_difference = received_phase - np.polyval(np.polyfit(t, received_phase, 1), t)

# FFT to detect vital frequencies
phase_fft = np.abs(fft(phase_difference))
frequencies = fftfreq(N, 1/fs)

# Positive frequencies
positive_freqs = frequencies[:N//2]
positive_fft = phase_fft[:N//2]

# Find Peaks
peaks, _ = find_peaks(positive_fft, height=np.max(positive_fft)*0.1)
peak_freqs = positive_freqs[peaks]

# Detected Vital Signs
breath_detected = None
heart_detected = None

for freq in peak_freqs:
    if 0.1 < freq < 0.5:
        breath_detected = freq * 60  # breaths/minute
    if 0.8 < freq < 2:
        heart_detected = freq * 60   # beats/minute

print(f"Detected Breath Rate: {breath_detected:.2f} breaths per minute")
print(f"Detected Heart Rate: {heart_detected:.2f} beats per minute")

# Plotting
plt.figure(figsize=(14,8))
plt.subplot(2,1,1)
plt.plot(t, phase_difference)
plt.title('Phase Difference due to Chest Motion')
plt.xlabel('Time (s)')
plt.ylabel('Phase (radians)')
plt.grid()

plt.subplot(2,1,2)
plt.plot(positive_freqs, positive_fft)
plt.title('FFT of Phase Difference')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 3)
plt.grid()

plt.tight_layout()
plt.show()

# ========== PART 2: COMMUNICATION using OFDM ==========

# Prepare Data (heart rate and breath rate) for transmission
# Example: continuously transmit heart and breath rate

def int_to_binary_list(value, bits=8):
    return list(map(int, np.binary_repr(int(value), width=bits)))

# BPSK Modulation: 0 -> -1, 1 -> +1
def bpsk_modulate(bits):
    return np.array([1 if bit == 1 else -1 for bit in bits])

# OFDM Parameters
num_subcarriers = 16
cp_len = 4  # Cyclic prefix length

def transmit_ofdm(heart_rate, breath_rate):
    # Convert heart rate and breath rate to binary
    heart_bits = int_to_binary_list(int(heart_rate), bits=8)
    breath_bits = int_to_binary_list(int(breath_rate), bits=8)
    data_bits = heart_bits + breath_bits  # total 16 bits

    # BPSK modulation
    bpsk_symbols = bpsk_modulate(data_bits)

    # IFFT (OFDM modulation)
    ofdm_time = ifft(bpsk_symbols)

    # Add cyclic prefix
    cyclic_prefix = ofdm_time[-cp_len:]
    ofdm_tx_signal = np.concatenate([cyclic_prefix, ofdm_time])

    return ofdm_tx_signal

# Simulate continuous transmission
for i in range(0, N, fs):  # Simulate over time (e.g., every second)
    heart_rate, breath_rate = generate_vital_signs(t[i:i+fs])  # Get updated heart and breath rate
    ofdm_tx_signal = transmit_ofdm(np.mean(heart_rate), np.mean(breath_rate))  # Transmit using OFDM

    # Plot transmitted OFDM signal for one window
    plt.figure(figsize=(10,4))
    plt.plot(np.real(ofdm_tx_signal))
    plt.title(f'Transmitted OFDM Signal (Real Part) at t={t[i]:.2f}s')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()