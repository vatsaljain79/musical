import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
        peak_scatter.set_offsets([])
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

if __name__ == "__main__":
    # Load a short audio snippet (10 sec for demo)
    y, sr = librosa.load("song1.mp3", sr=None, mono=True, duration=10)
    y /= np.max(np.abs(y))

    # Compute constellation + hashes
    mag, freqs, times = stft(y, sr)
    const_map = get_constellation_map(mag, freqs, times, prominence_db=15)
    hashes = generate_hashes(const_map, fan_out=3)

    # Animate (show live)
    animate_constellation(y, sr, const_map, hashes, save_path=None)

    # Or save to mp4/gif
    # animate_constellation(y, sr, const_map, hashes, save_path="constellation.mp4")
