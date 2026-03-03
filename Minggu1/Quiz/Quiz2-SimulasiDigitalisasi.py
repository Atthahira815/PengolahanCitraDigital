import numpy as np
import matplotlib.pyplot as plt

def simulate_digitization(analog_signal, sampling_rate, quantization_levels):
    # Sinyal analog kontinu
    x_cont = np.linspace(0, 1, 1000)
    y_cont = analog_signal(x_cont)

    # 1. Sampling
    x_sample = np.linspace(0, 1, sampling_rate)
    y_sample = analog_signal(x_sample)

    # 2. Kuantisasi
    y_min, y_max = y_sample.min(), y_sample.max()
    q_levels = np.linspace(y_min, y_max, quantization_levels)
    y_quantized_idx = np.digitize(y_sample, q_levels)
    y_quantized = q_levels[y_quantized_idx - 1]

    # 3. Visualisasi
    plt.figure(figsize=(10,5))
    plt.plot(x_cont, y_cont, label="Sinyal Analog", linewidth=2)
    plt.stem(x_sample, y_sample, linefmt='gray', markerfmt='o', basefmt=" ")
    plt.step(x_sample, y_quantized, where='mid', label="Sinyal Digital", color='red')
    plt.legend()
    plt.title("Simulasi Sampling dan Kuantisasi")
    plt.xlabel("Waktu")
    plt.ylabel("Amplitudo")
    plt.grid(alpha=0.3)
    plt.show()


# === PEMANGGILAN FUNGSI ===
simulate_digitization(
    analog_signal=lambda x: np.sin(2 * np.pi * 5 * x),
    sampling_rate=20,
    quantization_levels=8
)
