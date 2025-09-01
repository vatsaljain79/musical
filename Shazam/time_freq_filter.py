import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
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


def time_frequency_filter(magnitude, smoothing=(5, 5), threshold_db=6):
    """
    Apply time-frequency filtering:
    1. Estimate noise floor (median over time per frequency bin).
    2. Build a mask where signal > noise + threshold.
    3. Smooth mask with median filter.
    """
    mag_db = 20 * np.log10(magnitude + 1e-10)

    # Estimate noise floor (per frequency bin)
    noise_floor = np.median(mag_db, axis=1, keepdims=True)

    # Mask bins significantly above noise
    mask = mag_db > (noise_floor + threshold_db)

    # Median filter the mask to remove isolated speckles
    mask = median_filter(mask.astype(float), size=smoothing)

    # Apply mask back to magnitude
    attenuation = 0
    filtered_magnitude = magnitude * (mask + attenuation*(1-mask))
    return filtered_magnitude


def get_constellation_map(magnitude, freq_bins, time_bins,
                          prominence_db=30, max_peaks=5, max_freq=4000):
    """
    Build a constellation list of peak tuples: (time, freq, prominence_db)
    """
    constellation = []
    # convert to dB for peak picking
    mag_db = 20 * np.log10(magnitude + 1e-10) 

    for t_idx, frame_db in enumerate(mag_db.T):
        peaks, props = find_peaks(frame_db, prominence=prominence_db)
        if len(peaks) == 0:
            continue

        prominences = props.get("prominences", np.zeros_like(peaks))

        # filter peaks by max_freq
        valid_mask = freq_bins[peaks] <= max_freq
        peaks = peaks[valid_mask]
        prominences = prominences[valid_mask]

        if len(peaks) == 0:
            continue

        # sort and take top ones
        sorted_idx = np.argsort(prominences)[::-1]
        top_idx = sorted_idx[:max_peaks]

        for si in top_idx:
            p = peaks[si]
            freq = freq_bins[p]
            prom_db = prominences[si]
            time = float(time_bins[t_idx])
            constellation.append((time, float(freq), float(prom_db)))

    return constellation


# ---- Example usage ----
if __name__ == "__main__":
    mp3_path = "/home/vibgyor/BTP/musical/recordings/pachtaogerecording.mp3"
    signal, sr = librosa.load(mp3_path, sr=None, mono=True)

    # Normalize
    signal = signal / np.max(np.abs(signal))

    # Parameters
    fft_size = 2048
    hop_size = 512

    # STFT
    magnitude, freq_bins, time_bins = stft(signal, sr, fft_size, hop_size)

    # ---- Time-frequency filtering ----
    filtered_magnitude = time_frequency_filter(magnitude,
                                               smoothing=(7, 7),
                                               threshold_db=6)

    # Build constellation map (on filtered spectrogram)
    constellation = get_constellation_map(filtered_magnitude, freq_bins, time_bins,
                                          prominence_db=30, max_peaks=2)

    # Unpack constellation into separate lists
    if constellation:
        times, freqs, prom = zip(*constellation)
    else:
        times, freqs, prom = [], [], []

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(20 * np.log10(filtered_magnitude + 1e-6), origin='lower', aspect='auto',
               extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]])
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Filtered STFT Spectrogram with Constellation Map")

    # Overlay constellation peaks
    plt.scatter(times, freqs, c="#ff0f00", cmap="viridis", s=10, marker='o')
    plt.show()
