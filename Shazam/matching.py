import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import defaultdict, Counter
import matplotlib.animation as animation



# ---------- STEP 1: STFT ----------
def stft(signal, sr, fft_size=2048, hop_size=512, window=np.hanning):
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


# ---------- STEP 2: CONSTELLATION MAP ----------
def get_constellation_map(magnitude, freq_bins, time_bins,
                          prominence_db=20, max_peaks=2):
    constellation = []
    mag_db = 20 * np.log10(magnitude + 1e-10)

    for t_idx, frame in enumerate(mag_db.T):
        peaks, props = find_peaks(frame, prominence=prominence_db)
        if len(peaks) > 0:
            prominences = props["prominences"]
            sorted_idx = np.argsort(prominences)[::-1]
            top_peaks = peaks[sorted_idx[:max_peaks]]

            for p in top_peaks:
                freq = freq_bins[p]
                time = time_bins[t_idx]
                constellation.append((time, freq))

    return np.array(constellation)


# ---------- STEP 3: HASHING ----------
def generate_hashes(constellation, fan_out=5):
    """
    Create hashes from constellation map:
    (f1, f2, Δt) → anchor time
    """
    hashes = []
    for i in range(len(constellation)):
        t1, f1 = constellation[i]
        for j in range(1, fan_out + 1):
            if i + j < len(constellation):
                t2, f2 = constellation[i + j]
                dt = t2 - t1
                if 0 < dt < 5.0:  # limit Δt window
                    hash_val = (int(f1), int(f2), round(dt, 2))
                    hashes.append((hash_val, round(t1,2)))
    return hashes


# ---------- STEP 4: DATABASE ----------
class FingerprintDB:
    def __init__(self):
        self.db = defaultdict(list)  # (f1,f2,dt) → [(song_id, t_anchor)]

    def add_song(self, song_id, hashes):
        for h, t in hashes:
            self.db[h].append((song_id, t))

    def match(self, query_hashes):
        matches = []
        for h, qt in query_hashes:
            if h in self.db:
                for song_id, st in self.db[h]:
                    offset = round(st - qt, 2)  # time alignment
                    matches.append((song_id, offset))
        return matches


# ---------- STEP 5: SONG IDENTIFICATION ----------
def identify_song(db, query_hashes):
    matches = db.match(query_hashes)
    if not matches:
        return None

    counter = Counter(matches)
    best_match, votes = counter.most_common(1)[0]
    song_id, offset = best_match
    return song_id, votes



def animate_constellation(signal, sr, constellation, hashes, save_path=None, interval=200):
    """
    Animate constellation map + hashes.
    - signal: waveform
    - sr: sample rate
    - constellation: [(time, freq), ...]
    - hashes: [(hash_val, t_anchor), ...]
    - save_path: if given, save animation as mp4/gif
    - interval: ms between frames
    """

    # Spectrogram background
    S = librosa.stft(signal, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis='time', y_axis='log',
                             cmap='magma', ax=ax)

    ax.set_title("Animated Constellation Map + Hashes")
    ax.set_ylim(100, sr // 2)

    # Initialize plot elements
    peak_scatter = ax.scatter([], [], c='cyan', marker='.', s=20, label="Peaks")
    arrows = []
    ax.legend(loc="upper right")

    # Data prep
    times = [t for (t, f) in constellation]
    freqs = [f for (t, f) in constellation]

    def init():
        peak_scatter.set_offsets(np.empty((0,2)))
        return [peak_scatter]

    def update(frame_idx):
        # Show peaks up to current frame
        peak_scatter.set_offsets(np.column_stack((times[:frame_idx+1],
                                                  freqs[:frame_idx+1])))

        # Add arrows for hashes whose anchor time <= this frame
        while len(arrows) < len(hashes) and hashes[len(arrows)][1] <= times[frame_idx]:
            h, t1 = hashes[len(arrows)]
            f1, f2, dt = h
            t2 = t1 + dt
            arr = ax.arrow(t1, f1, (t2 - t1), (f2 - f1),
                           head_width=50, head_length=0.1,
                           color="yellow", alpha=0.6, length_includes_head=True)
            arrows.append(arr)

        return [peak_scatter] + arrows

    ani = animation.FuncAnimation(fig, update, frames=len(constellation),
                                  init_func=init, blit=False, interval=interval,
                                  repeat=False)

    if save_path:
        ani.save(save_path, fps=5, dpi=150)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return ani

# ---------- DEMO ----------
if __name__ == "__main__":
    # Database
    db = FingerprintDB()

    # Songs to index (replace with your mp3s)
    songs = {
        "song1": "music/Tujhe_Dekha_Toh.mp3",
        "song2": "music/Dheere_Dheere.mp3",
        "song3": "music/6_AM.mp3",
        "song4": "music/Agar_Tum_Saath_Ho.mp3",
        "song5": "music/Desi_Kalakaar.mp3",
        "song6": "music/Ho_Gya_Hai_Tujhko.mp3",
        # "song7": "recordings/trim_dheere.mp3"
    }

    # Index songs
    for song_id, path in songs.items():
        sig, sr = librosa.load(path, sr=None, mono=True)
        sig /= np.max(np.abs(sig))

        mag, freqs, times = stft(sig, sr)
        const_map = get_constellation_map(mag, freqs, times)
        hashes = generate_hashes(const_map)
        db.add_song(song_id, hashes)
        print(f"Indexed {song_id} with {len(hashes)} hashes")

    # Query (snippet of song1)
    query_sig, sr = librosa.load("recordings/trim_dheere.mp3", sr=None, mono=True)
    query_sig /= np.max(np.abs(query_sig))

    mag, freqs, times = stft(query_sig, sr)
    const_map = get_constellation_map(mag, freqs, times)
    query_hashes = generate_hashes(const_map)
    print(f"Indexed query song with {len(query_hashes)} hashes")

    result = identify_song(db, query_hashes)
    if result:
        print(f"Best match: {result[0]} with {result[1]} votes")
    else:
        print("No match found")

    # Animate (show live)
    # animate_constellation(query_sig, sr, const_map, hashes, save_path=None)

    # Or save to mp4/gif
    # animate_constellation(y, sr, const_map, hashes, save_path="constellation.mp4")

