import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from scipy.signal import chirp
from scipy.fft import fft, fftfreq, ifft
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import gridspec

# ========== PARAMETERS ==========
fs = 1000
T = 10
N = fs * T
f0, f1 = 20, 200
t = np.linspace(0, T, N)

# ========== SIGNAL GENERATION ==========
def generate_vital_signs(t):
    heart_rate = 60 + 10 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 2, len(t))
    breath_rate = 15 + 2 * np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 0.5, len(t))
    return heart_rate, breath_rate

# ========== GUI SETUP ==========
root = tk.Tk()
root.title("Hospital Vital Sign Monitor")

heart_rate_label = tk.Label(root, text="Heart Rate: -- bpm", font=("Helvetica", 36), fg="black")
heart_rate_label.pack(pady=20)

breath_rate_label = tk.Label(root, text="Breath Rate: -- bpm", font=("Helvetica", 36), fg="black")
breath_rate_label.pack(pady=20)

heart_rate_status_label = tk.Label(root, text="Heart Rate Status: --", font=("Helvetica", 24), fg="black")
heart_rate_status_label.pack(pady=10)

breath_rate_status_label = tk.Label(root, text="Breath Rate Status: --", font=("Helvetica", 24), fg="black")
breath_rate_status_label.pack(pady=10)

# ========== PLOT SETUP ==========
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[2, 0.2, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[2])
ax = [ax0, ax1]

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# ========== UPDATE FUNCTION ==========
def update_vital_signs_and_plot():
    heart_rate, breath_rate = generate_vital_signs(t)
    
    breath_motion = 0.05 * np.sin(2 * np.pi * breath_rate * t) + 0.02 * np.random.normal(size=len(t))
    heart_motion = 0.01 * np.sin(2 * np.pi * heart_rate * t) + 0.01 * np.random.normal(size=len(t))
    total_motion = breath_motion + heart_motion

    transmitted_signal = chirp(t, f0=f0, f1=f1, t1=T, method='linear')
    received_signal = transmitted_signal * np.exp(1j * 2 * np.pi * total_motion)

    received_phase = np.unwrap(np.angle(received_signal))
    phase_difference = received_phase - np.polyval(np.polyfit(t, received_phase, 1), t)

    phase_fft = np.abs(fft(phase_difference))
    frequencies = fftfreq(N, 1/fs)
    positive_freqs = frequencies[:N//2]
    positive_fft = phase_fft[:N//2]

    breath_band = (positive_freqs > 0.1) & (positive_freqs < 0.5)
    heart_band = (positive_freqs > 0.8) & (positive_freqs < 2.0)

    breath_peak_idx = np.argmax(positive_fft * breath_band)
    heart_peak_idx = np.argmax(positive_fft * heart_band)

    breath_detected = positive_freqs[breath_peak_idx] * 60 if breath_band[breath_peak_idx] else 0
    heart_detected = positive_freqs[heart_peak_idx] * 60 if heart_band[heart_peak_idx] else 0

    # Update GUI
    heart_rate_label.config(text=f"Heart Rate: {heart_detected:.2f} bpm")
    breath_rate_label.config(text=f"Breath Rate: {breath_detected:.2f} bpm")

    heart_status, heart_color = "Normal", "green"
    if heart_detected < 50:
        heart_status, heart_color = "Low", "red"
    elif heart_detected > 100:
        heart_status, heart_color = "High", "orange"
    heart_rate_status_label.config(text=f"Heart Rate Status: {heart_status}", fg=heart_color)

    breath_status, breath_color = "Normal", "green"
    if breath_detected < 12:
        breath_status, breath_color = "Low", "red"
    elif breath_detected > 20:
        breath_status, breath_color = "High", "orange"
    breath_rate_status_label.config(text=f"Breath Rate Status: {breath_status}", fg=breath_color)

    # ========== UPDATE PLOTS ==========
    ax[0].cla()
    step = 100  # downsample for visibility
    ax[0].plot(t[::step], breath_rate[::step], label="Breath Rate (Simulated)", alpha=0.7)
    ax[0].plot(t[::step], heart_rate[::step], label="Heart Rate (Simulated)", alpha=0.7)
    ax[0].set_title("Heart & Breath Rate (Simulated)")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Rate (bpm)")
    ax[0].legend()
    ax[0].set_xlim(t[0], t[-1])
    ax[0].set_xticks(np.linspace(0, t[-1], 11))
    ax[0].grid(True, linestyle='--', linewidth=0.5)

    ax[1].cla()
    ax[1].plot(positive_freqs, positive_fft)
    ax[1].axvspan(0.1, 0.5, color='red', alpha=0.2, label='Breath Band')
    ax[1].axvspan(0.8, 2.0, color='green', alpha=0.2, label='Heart Band')
    ax[1].set_title("FFT of Phase Difference (Detected)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude")
    ax[1].legend()
    ax[1].set_xlim(0, 3)
    ax[1].grid(True, linestyle='--', linewidth=0.5)

    canvas.draw()
    root.after(1000, update_vital_signs_and_plot)

# Start updating and run the app
update_vital_signs_and_plot()
root.mainloop()
