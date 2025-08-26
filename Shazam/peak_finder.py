import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import librosa

def stft(signal, sr, fft_size=2048, hop_size=512, window=np.hanning):
    """
    Compute Short-Time Fourier Transform (STFT) manually.
    """
    win = window(fft_size)
    num_frames = 1 + (len(signal) - fft_size) // hop_size
    stft_matrix = np.zeros((fft_size // 2 + 1, num_frames), dtype=np.complex64)

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + fft_size] * win
        spectrum = np.fft.rfft(frame)
        stft_matrix[:, i] = spectrum

    magnitude = np.abs(stft_matrix)
    freq_bins = np.fft.rfftfreq(fft_size, 1.0 / sr)
    time_bins = np.arange(num_frames) * hop_size / sr
    return magnitude, freq_bins, time_bins


def get_prominent_peaks(magnitude, freq_bins, prominence_db=20):
    """
    Get the most prominent peak in each STFT frame.
    """
    peak_frequencies = []
    mag_db = 20 * np.log10(magnitude + 1e-10)

    for frame in mag_db.T:  # iterate over time frames
        peaks, props = find_peaks(frame, prominence=prominence_db)
        if len(peaks) > 0:
            best_peak_idx = np.argmax(props["prominences"])
            best_freq = freq_bins[peaks[best_peak_idx]]
        else:
            best_freq = 0.0
        peak_frequencies.append(best_freq)

    return np.array(peak_frequencies)


# ---- Example usage ----
if __name__ == "__main__":
    mp3_path = "recordings_recording1.mp3"  # Change this to your file

    # Load MP3 (mono, float32)
    signal, sr = librosa.load(mp3_path, sr=None, mono=True)

    # Normalize
    signal = signal / np.max(np.abs(signal))

    # Parameters
    fft_size = 2048
    hop_size = 512

    # Compute STFT
    magnitude, freq_bins, time_bins = stft(signal, sr, fft_size, hop_size)

    # Find peaks
    peak_freqs = get_prominent_peaks(magnitude, freq_bins, prominence_db=20)

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(20 * np.log10(magnitude + 1e-6), origin='lower', aspect='auto',
               extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]])
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT Spectrogram")
    plt.show()

    # Plot peak frequency over time
    plt.figure(figsize=(10, 4))
    plt.plot(time_bins, peak_freqs, 'r')
    plt.xlabel("Time (s)")
    plt.ylabel("Prominent Peak Frequency (Hz)")
    plt.title("Most Prominent Peak per Frame (Noise-Resistant)")
    plt.show()
