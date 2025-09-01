import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D


def stft(signal, sr, fft_size=2048, hop_size=512):
    """
    Compute STFT magnitude with frequency and time bins.
    """
    stft_matrix = librosa.stft(signal, n_fft=fft_size, hop_length=hop_size, window="hann")
    magnitude = np.abs(stft_matrix)
    freq_bins = np.linspace(0, sr / 2, 1 + fft_size // 2)
    time_bins = np.arange(magnitude.shape[1]) * hop_size / sr
    return magnitude, freq_bins, time_bins


def get_constellation_map(magnitude, freq_bins, time_bins,
                          prominence_db=30, max_peaks=5, max_freq=4000):
    """
    Build a constellation list of peak tuples: (time, freq, prominence_db).
    """
    constellation = []
    # convert to dB for peak picking
    mag_db = 20 * np.log10(magnitude + 1e-10)

    for t_idx, frame_db in enumerate(mag_db.T):
        # find all peaks on dB-scaled frame
        peaks, props = find_peaks(frame_db, prominence=prominence_db)
        if len(peaks) == 0:
            continue

        prominences = props.get("prominences", np.zeros_like(peaks))

        # filter peaks by max_freq BEFORE selecting top ones
        valid_mask = freq_bins[peaks] <= max_freq
        peaks = peaks[valid_mask]
        prominences = prominences[valid_mask]

        if len(peaks) == 0:
            continue

        # sort remaining peaks by prominence desc and pick top ones
        sorted_idx = np.argsort(prominences)[::-1]
        top_idx = sorted_idx[:max_peaks]

        for si in top_idx:
            p = peaks[si]
            freq = freq_bins[p]
            prom_db = prominences[si]
            time = float(time_bins[t_idx])
            constellation.append((time, float(freq), float(prom_db)))

    return constellation


if __name__ == "__main__":
    mp3_path = "/home/vibgyor/BTP/musical/recordings/Pachtaogetrim.mp3"  # replace with your file
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
                                          prominence_db=30, max_peaks=2)

    # Unpack constellation into separate lists
    if constellation:
        times, freqs, prom = zip(*constellation)
    else:
        times, freqs, prom = [], [], []

    # --- 2D spectrogram with constellation overlay ---
    plt.figure(figsize=(10, 6))
    plt.imshow(20 * np.log10(magnitude + 1e-6), origin='lower', aspect='auto',
               extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]])
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT Spectrogram with Constellation Map")
    plt.scatter(times, freqs, c=prom, cmap="viridis", s=10, marker='o')
    plt.show()

    # --- 3D constellation plot ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(times, freqs, prom, c=prom, cmap="plasma", s=15)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel("Prominence (dB)")
    ax.set_title("3D Constellation Map")
    fig.colorbar(sc, label="Prominence (dB)")

    plt.show()
