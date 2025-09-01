import math
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import defaultdict, Counter
import matplotlib.animation as animation

# ---------- STEP 1: STFT ----------
def stft(signal, sr, fft_size=2048, hop_size=512, window=np.hanning):
    """
    Simple STFT that pads the end so partial last frames are included.
    Returns: magnitude (linear), freq_bins (Hz), time_bins (s)
    """
    win = window(fft_size)
    if len(signal) <= fft_size:
        num_frames = 1
    else:
        num_frames = 1 + int(math.ceil((len(signal) - fft_size) / float(hop_size)))

    # ensure enough length to extract num_frames frames
    length_needed = (num_frames - 1) * hop_size + fft_size
    if length_needed > len(signal):
        pad_len = length_needed - len(signal)
        signal = np.pad(signal, (0, pad_len), mode='constant')

    stft_matrix = np.zeros((fft_size // 2 + 1, num_frames), dtype=np.complex64)

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + fft_size] * win
        spectrum = np.fft.rfft(frame)
        stft_matrix[:, i] = spectrum

    magnitude = np.abs(stft_matrix)
    freq_bins = np.fft.rfftfreq(fft_size, 1.0 / sr)
    time_bins = np.arange(num_frames) * hop_size / float(sr)
    return magnitude, freq_bins, time_bins


# ---------- STEP 2: CONSTELLATION MAP ----------
def get_constellation_map(magnitude, freq_bins, time_bins,
                          prominence_db=20, max_peaks=2, max_freq=4000):
    """
    Build a constellation list of peak tuples: (time, freq, prominence_db)

    Parameters
    ----------
    magnitude : 2D array
        Linear magnitude matrix from stft (shape [freq_bins, time_bins])
    freq_bins, time_bins : arrays
        Frequency and time arrays returned by stft
    prominence_db : float
        Threshold for peak prominence in dB
    max_peaks : int
        Pick up to this many peaks per frame (by prominence)
    max_freq : float
        Ignore peaks above this frequency (Hz)

    Returns
    -------
    constellation : list of tuples
        (time, freq, prom_db)
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



# ---------- HELPERS (for hashing) ----------
def _detect_freq_unit(constellation):
    # returns "hz" if values look like Hz (max > 3000), else "bin"
    maxf = 0.0
    for entry in constellation:
        if len(entry) >= 2:
            f = entry[1]
            if f and f > maxf:
                maxf = f
    return "hz" if maxf > 3000 else "bin"

def _hz_to_semitone(f, ref=440.0):
    return 12 * math.log2(f / ref) if f > 0 else 0.0

def _quantize_freq(f, mode="bin", f_bits=14, semitone=False):
    # returns integer quantized frequency (0 .. (1<<f_bits)-1)
    max_val = (1 << f_bits) - 1
    if mode == "bin":
        q = int(round(f))
    else:  # mode == "hz"
        if semitone:
            semis = _hz_to_semitone(f)
            # center semitone range in the integer space to allow negatives
            q = int(round(semis)) + (1 << (f_bits - 1))
        else:
            q = int(round(f))
    return max(0, min(max_val, q))

def _quantize_dt(dt, max_dt=5.0, buckets=4096):
    # linear quantization of dt into buckets
    if dt <= 0:
        return 0
    b = int(math.floor((dt / max_dt) * (buckets - 1)))
    return max(0, min(buckets - 1, b))

def _quantize_amp(a, amp_bits=6):
    # quantize amplitude/prominence into amp_bits
    if a is None:
        return 0
    maxv = (1 << amp_bits) - 1
    # prominence is in dB typically; round and clamp
    q = int(round(a))
    return max(0, min(maxv, q))

def _pack_hash(f1q, f2q, dtq, ampq=0,
               f_bits=14, dt_bits=12, amp_bits=6):
    """
    pack bits: [f1 (f_bits)] [f2 (f_bits)] [dt (dt_bits)] [amp (amp_bits)]
    returns integer
    """
    shift_f2 = dt_bits + amp_bits
    shift_f1 = f_bits + shift_f2
    packed = (f1q << shift_f1) | (f2q << shift_f2) | (dtq << amp_bits) | ampq
    return packed


# ---------- STEP 3: HASHING ----------
def generate_hashes(constellation, fan_out=5):
    """
    Create hashes from constellation map:
    (f1, f2, Δt) → anchor time

    Produces multi-resolution packed integer hashes for robustness:
      - fine resolution
      - coarse resolution

    Each returned item has the shape: (hash_key, t_anchor)
    where hash_key is a tuple: (packed_hash_int, dt_rounded)
    This keeps equality semantics (same structure for indexing & querying)
    and lets animate_constellation access dt easily for visualization.
    """
    hashes = []
    if not constellation:
        return hashes

    # detect whether frequency values look like Hz or bin indices
    freq_mode = _detect_freq_unit(constellation)
    use_semitone = (freq_mode == "hz")  # use semitone conversion for Hz to gain pitch invariance

    # bit allocations
    # fine: higher resolution
    fine_f_bits = 14   # supports up to ~16k distinct freq bins/coded semitone range
    fine_dt_bits = 12  # 4096 dt buckets across max_dt -> ~0.0012s resolution for max_dt=5s
    amp_bits = 6       # amplitude/prominence quantization bits

    # coarse: lower resolution (more tolerant)
    coarse_f_bits = 10
    coarse_dt_bits = 10

    max_dt = 5.0

    for i in range(len(constellation)):
        entry1 = constellation[i]
        t1 = float(entry1[0])
        f1 = float(entry1[1])
        amp1 = float(entry1[2]) if len(entry1) > 2 else None

        for j in range(1, fan_out + 1):
            if i + j >= len(constellation):
                break
            entry2 = constellation[i + j]
            t2 = float(entry2[0])
            f2 = float(entry2[1])
            dt = t2 - t1
            if not (0 < dt < max_dt):
                continue

            # fine quantization
            f1q = _quantize_freq(f1, mode=freq_mode, f_bits=fine_f_bits, semitone=use_semitone)
            f2q = _quantize_freq(f2, mode=freq_mode, f_bits=fine_f_bits, semitone=use_semitone)
            dtq  = _quantize_dt(dt, max_dt=max_dt, buckets=(1 << fine_dt_bits))
            ampq = _quantize_amp(amp1, amp_bits=amp_bits)

            fine_hash_int = _pack_hash(f1q, f2q, dtq, ampq,
                                       f_bits=fine_f_bits, dt_bits=fine_dt_bits, amp_bits=amp_bits)
            # store dt (rounded) alongside packed int so visualizer can recover dt quickly
            fine_hash_key = (fine_hash_int, round(dt, 2))
            hashes.append((fine_hash_key, round(t1, 2)))

            # coarse quantization (more tolerant)
            f1q_c = _quantize_freq(f1, mode=freq_mode, f_bits=coarse_f_bits, semitone=use_semitone)
            f2q_c = _quantize_freq(f2, mode=freq_mode, f_bits=coarse_f_bits, semitone=use_semitone)
            dtq_c  = _quantize_dt(dt, max_dt=max_dt, buckets=(1 << coarse_dt_bits))
            ampq_c = _quantize_amp(amp1, amp_bits=min(amp_bits, 4))

            coarse_hash_int = _pack_hash(f1q_c, f2q_c, dtq_c, ampq_c,
                                         f_bits=coarse_f_bits, dt_bits=coarse_dt_bits, amp_bits=min(amp_bits,4))
            coarse_hash_key = (coarse_hash_int, round(dt, 2))
            hashes.append((coarse_hash_key, round(t1, 2)))

    return hashes


# ---------- STEP 4: DATABASE ----------
class FingerprintDB:
    def __init__(self):
        # keys are hash keys (tuple: (packed_int, dt_rounded)); values are list of (song_id, t_anchor)
        self.db = defaultdict(list)

    def add_song(self, song_id, hashes):
        """
        hashes: list of (hash_key, t_anchor)
        where hash_key is the same structure produced by generate_hashes
        """
        for h, t in hashes:
            self.db[h].append((song_id, t))

    def match(self, query_hashes):
        """
        query_hashes: list of (hash_key, t_query)
        returns list of (song_id, offset) for all matches
        offset = stored_anchor_time - query_time (rounded)
        """
        matches = []
        for h, qt in query_hashes:
            if h in self.db:
                for song_id, st in self.db[h]:
                    offset = round(st - qt, 2)  # time alignment
                    matches.append((song_id, offset))
        return matches


# ---------- STEP 5: SONG IDENTIFICATION ----------
def identify_song(db, query_hashes):
    """
    Given a FingerprintDB instance and query hashes, return best matched (song_id, votes)
    Votes is the count of matching (song_id, offset) pairs for the winning offset alignment.
    Returns None if no matches.
    """
    matches = db.match(query_hashes)
    if not matches:
        return None

    # Count by exact (song_id, offset) to reward consistent alignment
    counter = Counter(matches)
    (best_song_offset, votes) = counter.most_common(1)[0]  # ((song_id, offset), votes)
    song_id, offset = best_song_offset
    return song_id, votes


# ---------- Visualizer ----------
def animate_constellation(signal, sr, constellation, hashes, save_path=None, interval=200):
    """
    Animate constellation map + hashes.
    - constellation: list of (time, freq, prom_db)
    - hashes: list of (hash_key, t_anchor) where hash_key may be (packed_int, dt_rounded)
    Visualizer recovers f1,f2 by finding nearest peaks in constellation for times t_anchor and t_anchor+dt.
    """
    # Spectrogram background (librosa)
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
    times = np.array([t for (t, f, *rest) in constellation])
    freqs = np.array([f for (t, f, *rest) in constellation])

    def init():
        peak_scatter.set_offsets(np.empty((0, 2)))
        return [peak_scatter]

    def _find_freq_at_time(query_t, tol=0.05):
        # find nearest index by time; require within tol seconds to accept
        if len(times) == 0:
            return None
        idx = int(np.argmin(np.abs(times - query_t)))
        if abs(times[idx] - query_t) <= tol:
            return freqs[idx]
        return None

    def update(frame_idx):
        # Show peaks up to current frame
        if frame_idx >= len(times):
            idx_lim = len(times) - 1
        else:
            idx_lim = frame_idx
        if idx_lim >= 0:
            peak_scatter.set_offsets(np.column_stack((times[:idx_lim+1],
                                                      freqs[:idx_lim+1])))

        # Add arrows for hashes whose anchor time <= this frame's time
        # we assume hashes sorted by t_anchor; if not, sort them first
        # but to avoid re-sorting here, we will iterate hashes in order and add arrows when anchor <= current time
        current_time = times[min(frame_idx, len(times)-1)] if len(times) > 0 else 0.0
        while len(arrows) < len(hashes) and hashes[len(arrows)][1] <= current_time:
            h_key, t1 = hashes[len(arrows)]
            # if key is tuple (packed_int, dt_rounded), extract dt_rounded
            if isinstance(h_key, tuple) and len(h_key) >= 2:
                dt = float(h_key[1])
            else:
                # no dt available — skip drawing this arrow
                dt = None

            f1 = _find_freq_at_time(t1)
            if dt is None or f1 is None:
                # can't draw arrow without dt and f1
                arrows.append(None)
                continue

            t2 = t1 + dt
            f2 = _find_freq_at_time(t2)
            if f2 is None:
                # try a larger tolerance
                f2 = _find_freq_at_time(t2, tol=0.15)

            if f2 is None:
                arrows.append(None)
                continue

            arr = ax.arrow(t1, f1, (t2 - t1), (f2 - f1),
                           head_width=50, head_length=0.05,
                           color="yellow", alpha=0.6, length_includes_head=True)
            arrows.append(arr)

        # filter out None elements for return
        artists = [peak_scatter] + [a for a in arrows if a is not None]
        return artists

    ani = animation.FuncAnimation(fig, update, frames=max(1, len(times)),
                                  init_func=init, blit=False, interval=interval,
                                  repeat=False)

    if save_path:
        ani.save(save_path, fps=5, dpi=150)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return ani


# ---------- DEMO (keep as example) ----------
if __name__ == "__main__":
    # Database
    db = FingerprintDB()

    # Songs to index (replace with your mp3s)
    songs = {
        "song1": "/home/vibgyor/BTP/musical/music/Tujhe_Dekha_Toh.mp3",
        "song2": "/home/vibgyor/BTP/musical/music/Dheere_Dheere.mp3",
        "song3": "/home/vibgyor/BTP/musical/music/6_AM.mp3",
        "song4": "/home/vibgyor/BTP/musical/music/Alag_aasman.mp3",
        "song5": "/home/vibgyor/BTP/musical/music/Jeena_Jeena.mp3",
        "song6": "/home/vibgyor/BTP/musical/music/Chaar_kadam.mp3",
        "song7": "/home/vibgyor/BTP/musical/music/Chaand_Baaliyan.mp3",
        "song9": "/home/vibgyor/BTP/musical/music/Pachtaoge.mp3",
    }

    # Index songs
    for song_id, path in songs.items():
        sig, sr = librosa.load(path, sr=None, mono=True)
        sig = sig / (np.max(np.abs(sig)) + 1e-12)

        mag, freqs, times = stft(sig, sr)
        const_map = get_constellation_map(mag, freqs, times)
        hashes = generate_hashes(const_map)
        db.add_song(song_id, hashes)
        print(f"Indexed {song_id} with {len(hashes)} hashes")

    # Query (snippet of song1)
    query_sig, sr = librosa.load("/home/vibgyor/BTP/musical/recordings/chaar_kadam_recording.mp3", sr=None, mono=True)
    query_sig = query_sig / (np.max(np.abs(query_sig)) + 1e-12)

    mag, freqs, times = stft(query_sig, sr)
    const_map = get_constellation_map(mag, freqs, times)
    query_hashes = generate_hashes(const_map)
    print(f"Indexed query song with {len(query_hashes)} hashes")

    result = identify_song(db, query_hashes)
    if result:
        print(f"Best match: {result[0]} with {result[1]} votes")
    else:
        print("No match found")

    # animate_constellation(query_sig, sr, const_map, query_hashes, save_path=None)
