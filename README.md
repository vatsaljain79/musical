# Musical Note Recognition – Audio Recording & YouTube to MP3 Web App

This is a Flask web application with two main features:
1. **🎤 Audio Recorder** – Record audio from your browser and save it as an `.mp3`.
2. **🎵 YouTube to MP3** – Download audio from a YouTube link and save it in `.mp3` format.

---

## Features

### Audio Recorder
- 🎤 Start/stop recording from the browser (max 20 seconds).
- ⏱ Live timer while recording.
- 💾 Automatically names files as `recording1.mp3`, `recording2.mp3`, etc., if no filename is provided.
- 🔄 If a filename exists, appends `_1`, `_2`, etc., to avoid overwriting.
- 📥 Download the recorded audio directly from the browser.

### YouTube to MP3
- 📺 Paste any YouTube music/video link.
- 🎵 Extracts audio and saves it as `.mp3` with the video’s title.
- 📂 All downloaded songs are stored in the **music/** folder.
- ✅ Shows download link immediately below the input box without leaving the page.

---

## Requirements
- **Python 3.8+**
- **pip** (Python package manager)
- **ffmpeg** (installed and added to PATH)
- **yt-dlp** (for YouTube audio extraction)

---

## Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/vatsaljain79/Musical-Note-Recognition.git
cd Musical-Note-Recognition
```

### 2️⃣ Install dependencies
```bash
pip install flask pydub yt-dlp
```

### 3️⃣ Install FFmpeg
1. Download FFmpeg: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract the zip.
3. Locate the `bin` folder (contains `ffmpeg.exe`).
4. Add it to **System PATH**:
   - Press `Windows` key → type `Environment Variables` → click **Edit the system environment variables**.
   - Click **Environment Variables**.
   - Under **System variables**, select `Path` → **Edit** → **New** → paste the `bin` path.
   - Click **OK** on all dialogs.
5. Verify:
```bash
ffmpeg -version
```

---

## Running the App
```bash
python app.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Usage

### 🎤 Audio Recorder
1. Go to the **"Record Audio"** page.
2. Click **Start Recording**.
3. Speak or play music.
4. Click **Stop Recording** or let it auto-stop after 20 seconds.
5. Name your file (or leave blank for auto-generated name).
6. Download the `.mp3`.

### 🎵 YouTube to MP3
1. Go to the **"YouTube to MP3"** page.
2. Paste your YouTube link.
3. Click **Download**.
4. Once ready, a download link appears below.
5. File is saved in the **music/** folder.

---

## Project Structure
```
Musical-Note-Recognition/
│
├── app.py                 # Flask backend
├── templates/
│   ├── index.html          # Audio recorder page
│   ├── youtube.html        # YouTube to MP3 page
├── recordings/             # Recorded audio files
├── music/                  # YouTube downloaded MP3 files
└── README.md
```

---

## Notes
- Works only in browsers that support the `MediaRecorder` API.
- If microphone access is blocked, enable it in browser settings.
- For YouTube downloads, ensure your internet connection is active.
- Large YouTube videos may take time to process.

---

## License
This project is for educational purposes only. Feel free to modify and use.
