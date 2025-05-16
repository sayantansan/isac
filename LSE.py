import numpy as np
import matplotlib.pyplot as plt

# Parameters for sinusoidal signal
amplitude = 1  # Amplitude of the sinusoidal signal
frequency = 0.1  # Frequency of the sinusoidal signal
phase = 0  # Phase of the sinusoidal signal
sequence_length = 100  # Length of the sequence
sampling_rate = 1  # Sampling rate (samples per unit time)
time = np.arange(0, sequence_length)  # Time samples

# Generate sinusoidal transmitted signal
transmitted_signal = amplitude * np.sin(2 * np.pi * frequency * time + phase)

# Generate Gaussian Channel Response (Actual h)
h_mean = 0
h_variance = 1
h_size = 100
h = np.random.normal(h_mean, np.sqrt(h_variance), h_size)

# Generate Gaussian noise
noise_mean = 0  
noise_variance = 1
gaussian_noise = np.random.normal(noise_mean, np.sqrt(noise_variance), sequence_length)

# Known Received Signal Y
y_mean = 1
y_variance = 1
y = np.random.normal(y_mean, np.sqrt(y_variance), sequence_length)

# MMSE Estimation (simplified)
def mmse_estimation(y, transmitted_signal, h, sequence_length):
    mean_squared_error = np.zeros(h_size)
    for i in range(h_size):
        mean_squared_error[i] = np.sum((y - np.convolve(transmitted_signal, h[i], mode="same"))**2) / len(y)
    min_error = np.min(mean_squared_error)
    H_mmse = h[np.argmin(mean_squared_error)]
    estimated_signal_mmse = np.convolve(transmitted_signal, H_mmse, mode="same") + gaussian_noise
    return H_mmse, estimated_signal_mmse

# MLE Estimation (simplified)
def mle_estimation(y, transmitted_signal, h):
    def likelihood(y, h_mean, h_variance):
        return (1 / (np.sqrt(2 * np.pi * h_variance))) * np.exp(-(y - h_mean)**2 / (2 * h_variance))
    function = likelihood(y, h_mean, h_variance)
    max_likelihood = np.max(function)
    H_mle = h[np.argmax(function)]
    estimated_signal_mle = np.convolve(transmitted_signal, H_mle, mode="same") + gaussian_noise
    return H_mle, estimated_signal_mle

# MAP Estimation (simplified)
def map_estimation(y, transmitted_signal, h):
    def prior(x, h_mean, h_variance):
        return (1 / (np.sqrt(2 * np.pi * h_variance))) * np.exp(-(x - h_mean)**2 / (2 * h_variance))
    
    # Calculate likelihood for each value in h
    likelihood_values = (1 / (np.sqrt(2 * np.pi * h_variance))) * np.exp(-(y - h)**2 / (2 * h_variance))
    
    # Calculate prior for each h value
    prior_values = prior(h, h_mean, h_variance)
    
    # Calculate the posterior as the product of likelihood and prior
    posterior_values = likelihood_values * prior_values
    
    # Find the maximum posterior value
    max_posterior_index = np.argmax(posterior_values)
    H_map = h[max_posterior_index]
    
    # Retrieve the signal using the estimated H_map
    estimated_signal_map = np.convolve(transmitted_signal, H_map, mode="same") + gaussian_noise
    return H_map, estimated_signal_map

# LSE Estimation (simplified)
def lse_estimation(y, transmitted_signal, h):
    squared_error = np.zeros(h_size)
    for i in range(h_size):
        squared_error[i] = np.sum((y - np.convolve(transmitted_signal, h[i], mode="same"))**2)
    least_error = np.min(squared_error)
    H_lse = h[np.argmin(squared_error)]
    estimated_signal_lse = np.convolve(transmitted_signal, H_lse, mode="same") + gaussian_noise
    return H_lse, estimated_signal_lse

# Perform Estimations
H_mmse, estimated_signal_mmse = mmse_estimation(y, transmitted_signal, h, sequence_length)
H_mle, estimated_signal_mle = mle_estimation(y, transmitted_signal, h)
H_map, estimated_signal_map = map_estimation(y, transmitted_signal, h)
H_lse, estimated_signal_lse = lse_estimation(y, transmitted_signal, h)

# Normalize the transmitted signal and estimated signals
def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# Normalize all signals
normalized_transmitted_signal = normalize_signal(transmitted_signal)
normalized_estimated_signal_mmse = normalize_signal(estimated_signal_mmse)
normalized_estimated_signal_mle = normalize_signal(estimated_signal_mle)
normalized_estimated_signal_map = normalize_signal(estimated_signal_map)
normalized_estimated_signal_lse = normalize_signal(estimated_signal_lse)

# Probability of error (MSE) calculation
def probability_of_error(signal1, signal2):
    return np.mean((signal1 - signal2)**2)

# Calculate Probability of Error (MSE) for all estimations
pe_mmse = probability_of_error(normalized_transmitted_signal, normalized_estimated_signal_mmse)
pe_mle = probability_of_error(normalized_transmitted_signal, normalized_estimated_signal_mle)
pe_map = probability_of_error(normalized_transmitted_signal, normalized_estimated_signal_map)
pe_lse = probability_of_error(normalized_transmitted_signal, normalized_estimated_signal_lse)

# Print Probability of Error (MSE) for each estimation method
print(f"Probability of Error (MMSE): {pe_mmse:.4f}")
print(f"Probability of Error (MLE): {pe_mle:.4f}")
print(f"Probability of Error (MAP): {pe_map:.4f}")
print(f"Probability of Error (LSE): {pe_lse:.4f}")

# SNR Calculation
def snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

snr_mmse = snr(estimated_signal_mmse, gaussian_noise)
snr_mle = snr(estimated_signal_mle, gaussian_noise)
snr_map = snr(estimated_signal_map, gaussian_noise)
snr_lse = snr(estimated_signal_lse, gaussian_noise)

# Plot Actual Transmitted Signal vs Retrieved Signals (MMSE, MLE, MAP, LSE)
plt.figure(figsize=(14, 12))

# MMSE Plot
plt.subplot(3, 2, 1)
plt.plot(time, transmitted_signal, label="Actual Transmitted Signal", color='blue')
plt.plot(time, estimated_signal_mmse, label="Retrieved Signal (MMSE)", linestyle='--', color='red')
plt.title("Actual vs Retrieved Signal (MMSE)")
plt.xlabel("Sample Index")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True)

# MLE Plot
plt.subplot(3, 2, 2)
plt.plot(time, transmitted_signal, label="Actual Transmitted Signal", color='blue')
plt.plot(time, estimated_signal_mle, label="Retrieved Signal (MLE)", linestyle='--', color='green')
plt.title("Actual vs Retrieved Signal (MLE)")
plt.xlabel("Sample Index")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True)

# MAP Plot
plt.subplot(3, 2, 3)
plt.plot(time, transmitted_signal, label="Actual Transmitted Signal", color='blue')
plt.plot(time, estimated_signal_map, label="Retrieved Signal (MAP)", linestyle='--', color='orange')
plt.title("Actual vs Retrieved Signal (MAP)")
plt.xlabel("Sample Index")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True)

# LSE Plot
plt.subplot(3, 2, 4)
plt.plot(time, transmitted_signal, label="Actual Transmitted Signal", color='blue')
plt.plot(time, estimated_signal_lse, label="Retrieved Signal (LSE)", linestyle='--', color='purple')
plt.title("Actual vs Retrieved Signal (LSE)")
plt.xlabel("Sample Index")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True)

# Plot for Actual h vs Estimated h
plt.subplot(3, 2, 5)
plt.plot(h, label="Actual h", color='blue')
plt.plot([H_mmse]*h_size, label="Estimated h (MMSE)", linestyle='--', color='red')
plt.plot([H_mle]*h_size, label="Estimated h (MLE)", linestyle='--', color='green')
plt.plot([H_map]*h_size, label="Estimated h (MAP)", linestyle='--', color='orange')
plt.plot([H_lse]*h_size, label="Estimated h (LSE)", linestyle='--', color='purple')
plt.title("Actual vs Estimated h")
plt.xlabel("Index")
plt.ylabel("h Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
