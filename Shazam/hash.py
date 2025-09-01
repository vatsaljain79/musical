import numpy as np
import librosa
from collections import defaultdict, Counter
from pathlib import Path


# ---------- STEP 1: PITCH TRACKING ----------
def extract_pitch(signal, sr, fmin=80.0, fmax=1000.0, frame_length=2048, hop_length=512):
    """
    Extract pitch contour using librosa's YIN algorithm.
    Returns times (s) and pitch contour (Hz).
    """
    pitches = librosa.yin(signal, fmin=fmin, fmax=fmax,
                          sr=sr, frame_length=frame_length, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(pitches)), sr=sr, hop_length=hop_length)
    return times, pitches


# ---------- STEP 2: SIMPLIFY MELODY ----------
def simplify_melody(times, pitches, min_duration=0.1):
    """
    Quantize and simplify the melody contour to stable notes.
    - times: frame times
    - pitches: pitch contour (Hz)
    Returns a list of (time, midi_note).
    """
    midi_notes = librosa.hz_to_midi(pitches)
    notes = []

    prev_note = None
    start_time = times[0]

    for t, note in zip(times, midi_notes):
        if np.isnan(note):
            continue
        note = int(round(note))
        if prev_note is None:
            prev_note = note
            start_time = t
            continue
        if note != prev_note:
            duration = t - start_time
            if duration >= min_duration:
                notes.append((start_time, prev_note))
            prev_note = note
            start_time = t

    if prev_note is not None:
        duration = times[-1] - start_time
        if duration >= min_duration:
            notes.append((start_time, prev_note))

    return notes


# ---------- STEP 3: FINGERPRINTING ----------
def create_address(anchor, target):
    """
    Create a 32-bit fingerprint address from anchor and target notes.
    Anchor = (time, midi), Target = (time, midi).
    """
    anchor_note = int(anchor[1])
    target_note = int(target[1])
    delta_time = int((target[0] - anchor[0]) * 1000)  # ms

    address = (anchor_note << 23) | (target_note << 14) | (delta_time & 0x3FFF)
    return np.uint32(address)


def fingerprint(notes, song_id, target_zone_size=5):
    """
    Generate fingerprints from a melody contour.
    Returns {address: (anchor_time_ms, song_id)}.
    """
    fingerprints = {}
    for i, anchor in enumerate(notes):
        for j in range(i + 1, min(i + 1 + target_zone_size, len(notes))):
            target = notes[j]
            address = create_address(anchor, target)
            anchor_time_ms = int(anchor[0] * 1000)
            fingerprints[address] = (anchor_time_ms, song_id)
    return fingerprints


# ---------- STEP 4: DATABASE ----------
class FingerprintDB:
    def __init__(self):
        self.db = defaultdict(list)

    def add_song(self, song_id, notes):
        fps = fingerprint(notes, song_id)
        for addr, couple in fps.items():
            self.db[addr].append(couple)

    def match(self, query_notes):
        query_fps = fingerprint(query_notes, "query")
        matches = []
        for addr, (qt, _) in query_fps.items():
            if addr in self.db:
                for st, song_id in self.db[addr]:
                    offset = st - qt
                    matches.append((song_id, offset))
        return matches


# ---------- STEP 5: SONG IDENTIFICATION ----------
def identify_song(db, query_notes):
    matches = db.match(query_notes)
    if not matches:
        return None
    counter = Counter(matches)
    (song_id, offset), votes = counter.most_common(1)[0]
    return song_id, votes


# ---------- DEMO ----------
if __name__ == "__main__":
    db = FingerprintDB()

    # Songs to index (replace with your paths)
    songs = {
        "song1": "/home/vibgyor/BTP/musical/music/Tujhe_Dekha_Toh.mp3",
        "song2": "/home/vibgyor/BTP/musical/music/Dheere_Dheere.mp3",
        "song3": "/home/vibgyor/BTP/musical/music/6_AM.mp3",
        "song4": "/home/vibgyor/BTP/musical/music/Agar_Tum_Saath_Ho.mp3",
        "song5": "/home/vibgyor/BTP/musical/music/Desi_Kalakaar.mp3",
        "song6": "/home/vibgyor/BTP/musical/music/Ho_Gya_Hai_Tujhko.mp3",
        "song7": "/home/vibgyor/BTP/musical/music/Pachtaoge.mp3",
        "song8": "/home/vibgyor/BTP/musical/music/Alag_aasman.mp3",
        "song9": "/home/vibgyor/BTP/musical/music/Jeena_Jeena.mp3",
        "song10": "/home/vibgyor/BTP/musical/music/Chaar_kadam.mp3",
        "song11": "/home/vibgyor/BTP/musical/music/Chaand_Baaliyan.mp3",
    }

    # Index songs
    for song_id, path in songs.items():
        sig, sr = librosa.load(path, sr=None, mono=True)
        sig /= np.max(np.abs(sig))
        times, pitches = extract_pitch(sig, sr)
        notes = simplify_melody(times, pitches)
        db.add_song(song_id, notes)
        print(f"Indexed {song_id} with {len(notes)} notes")

    # Query (voice recording of singing)
    query_path = Path("/home/vibgyor/BTP/musical/recordings/jeena_jeena_recording.mp3")
    qsig, sr = librosa.load(query_path, sr=None, mono=True)
    qsig /= np.max(np.abs(qsig))
    qtimes, qpitches = extract_pitch(qsig, sr)
    qnotes = simplify_melody(qtimes, qpitches)
    print(f"Query extracted {len(qnotes)} notes")

    result = identify_song(db, qnotes)
    if result:
        print(f"Best match: {result[0]} with {result[1]} votes")
    else:
        print("No match found")
