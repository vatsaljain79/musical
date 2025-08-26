import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import librosa


def stft(signal, sr, fft_size=2048, hop_size=512, window=np.hanning):
    """
    Compute Short-Time Fourier Transform (STFT).
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


def get_constellation_map(magnitude, freq_bins, time_bins, prominence_db=20, max_peaks=2):
    """
    Extract multiple prominent peaks per frame (constellation map).
    """
    constellation = []
    mag_db = 20 * np.log10(magnitude + 1e-10)

    for t_idx, frame in enumerate(mag_db.T):  # iterate over time frames
        peaks, props = find_peaks(frame, prominence=prominence_db)
        if len(peaks) > 0:
            # Sort peaks by prominence and take top-N
            prominences = props["prominences"]
            sorted_idx = np.argsort(prominences)[::-1]  # descending
            top_peaks = peaks[sorted_idx[:max_peaks]]

            for p in top_peaks:
                freq = freq_bins[p]
                time = time_bins[t_idx]
                constellation.append((time, freq))

    return np.array(constellation)


# ---- Example usage ----
if __name__ == "__main__":
    mp3_path = "recordings/Pachtaogerecording.mp3"  # replace with your file
    signal, sr = librosa.load(mp3_path, sr=None, mono=True)

    # Normalize
    signal = signal / np.max(np.abs(signal))

    # Parameters
    fft_size = 2048
    hop_size = 512

    # STFT
    magnitude, freq_bins, time_bins = stft(signal, sr, fft_size, hop_size)

    # Build constellation map
    constellation = get_constellation_map(magnitude, freq_bins, time_bins,
                                          prominence_db=20, max_peaks=5)

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(20 * np.log10(magnitude + 1e-6), origin='lower', aspect='auto',
               extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]])
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT Spectrogram with Constellation Map")

    # Overlay peaks
    plt.scatter(constellation[:, 0], constellation[:, 1], color='red', s=10, marker='o')
    plt.show()
