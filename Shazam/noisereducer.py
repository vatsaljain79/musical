import librosa
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment

# Step 1: Load audio
y, sr = librosa.load("recordings/Dheere_Dheere_Se_HrithikRoshan.mp3", sr=None)

# Step 2: Reduce noise
reduced_noise = nr.reduce_noise(y=y, sr=sr)

# Step 3: Save as temporary WAV
sf.write("temp.wav", reduced_noise, sr)

# Step 4: Convert WAV to MP3
sound = AudioSegment.from_wav("temp.wav")
sound.export("cleaned_output.mp3", format="mp3")
