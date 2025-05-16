import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Constants
SENSING_BITS = 30  # Percentage of bits used for sensing
COMMUNICATION_BITS = 70  # Percentage of bits used for communication
NOISE_STD_DEV_SENSE = 0.1  # Standard deviation for noise in sensing (more accuracy = smaller value)
NOISE_STD_DEV_COMM = 0.2  # Standard deviation for noise in communication (more noise = larger value)

# Simulate random object data (position, velocity)
def generate_object_data():
    # Random position (x, y in meters) and velocity (vx, vy in m/s)
    position = np.random.uniform(0, 10, 2)  # x, y position in meters
    velocity = np.random.uniform(0, 5, 2)  # vx, vy velocity in m/s
    return position, velocity

# Simulate Sensing (Add Gaussian Noise)
def sense_object(position, velocity, sensing_bits):
    # Add noise to sensed position and velocity to simulate imperfect sensing
    sensed_position = position + np.random.normal(0, NOISE_STD_DEV_SENSE, size=position.shape)
    sensed_velocity = velocity + np.random.normal(0, NOISE_STD_DEV_SENSE, size=velocity.shape)
    return sensed_position, sensed_velocity

# Simulate Communication (Add Gaussian Noise)
def communicate_data(position, velocity, communication_bits):
    # Combine position and velocity as transmitted signal
    transmitted_signal = np.concatenate((position, velocity))  # Shape: (4,)
    
    # Add noise to simulate communication imperfections
    noise = np.random.normal(0, NOISE_STD_DEV_COMM, transmitted_signal.shape)
    received_signal = transmitted_signal + noise
    return received_signal, transmitted_signal

# Calculate accuracy using Mean Squared Error and Signal-to-Noise Ratio (SNR)
def calculate_accuracy(transmitted_signal, received_signal):
    # Mean Squared Error (MSE)
    mse = mean_squared_error(transmitted_signal, received_signal)
    
    # Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(transmitted_signal**2)
    noise_power = np.mean((transmitted_signal - received_signal)**2)
    snr = 10 * np.log10(signal_power / noise_power)  # in dB
    
    return mse, snr

# Main simulation for ISAC
def run_isac_simulation():
    # Step 1: Generate random object data (position, velocity)
    position, velocity = generate_object_data()
    print(f"Generated Position: {position}")
    print(f"Generated Velocity: {velocity}")

    # Step 2: Sense the object (using sensing bits)
    sensed_position, sensed_velocity = sense_object(position, velocity, SENSING_BITS)
    print(f"Sensed Position: {sensed_position}")
    print(f"Sensed Velocity: {sensed_velocity}")

    # Step 3: Communicate the sensed data (using communication bits)
    received_signal, transmitted_signal = communicate_data(sensed_position, sensed_velocity, COMMUNICATION_BITS)
    print(f"Transmitted Signal: {transmitted_signal}")
    print(f"Received Signal: {received_signal}")

    # Step 4: Calculate accuracy (MSE and SNR)
    mse, snr = calculate_accuracy(transmitted_signal, received_signal)
    print(f"Accuracy (MSE between transmitted and received data): {mse}")
    print(f"Signal-to-Noise Ratio (SNR): {snr} dB")

    # Step 5: Plotting the results for Position and Velocity
    plt.figure(figsize=(12, 6))

    # Plot position data
    plt.subplot(1, 2, 1)
    plt.plot(position[0], position[1], 'go', label="Generated Position")
    plt.plot(sensed_position[0], sensed_position[1], 'bo', label="Sensed Position")
    plt.plot(received_signal[0], received_signal[1], 'ro', label="Received Position")
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Position Comparison')
    plt.legend()

    # Plot velocity data
    plt.subplot(1, 2, 2)
    plt.plot(velocity[0], velocity[1], 'bo', label="Generated Velocity")
    plt.plot(sensed_velocity[0], sensed_velocity[1], 'go', label="Sensed Velocity")
    plt.plot(received_signal[2], received_signal[3], 'yo', label="Received Velocity")
    plt.xlabel('X Velocity (m/s)')
    plt.ylabel('Y Velocity (m/s)')
    plt.title('Velocity Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the ISAC simulation
run_isac_simulation()
