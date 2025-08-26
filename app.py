import os
from flask import Flask, request, render_template, send_from_directory
from pydub import AudioSegment
import yt_dlp

app = Flask(__name__)


SAVE_DIR = "recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

MUSIC_DIR = "music"
os.makedirs(MUSIC_DIR, exist_ok=True)


def get_next_filename(base_name="recording", extension=".mp3"):
    counter = 1
    while True:
        filename = f"{base_name}{counter}{extension}"
        if not os.path.exists(os.path.join(SAVE_DIR, filename)):
            return filename
        counter += 1


def make_unique_filename(filename):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(SAVE_DIR, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1
    return new_filename


@app.route("/")
def index():
    return render_template("index.html", title="Record Audio")


@app.route("/save_audio", methods=["POST"])
def save_audio():
    audio_file = request.files["audio_data"]
    user_filename = request.form.get("filename", "").strip()

    if not user_filename:
        filename = get_next_filename()
    else:
        if not user_filename.lower().endswith(".mp3"):
            user_filename += ".mp3"
        filename = make_unique_filename(user_filename)

    temp_path = os.path.join(SAVE_DIR, "temp.webm")
    audio_file.save(temp_path)

    sound = AudioSegment.from_file(temp_path, format="webm")
    sound.export(os.path.join(SAVE_DIR, filename), format="mp3")
 
    os.remove(temp_path)

    return f"Saved as {filename} - <a href='/download/{filename}' target='_blank'>Download</a>"


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(SAVE_DIR, filename, as_attachment=True)


@app.route("/youtube")
def youtube_page():
    return render_template("youtube.html", title="YouTube Download")


@app.route("/download_youtube", methods=["POST"])
def download_youtube():
    youtube_url = request.form.get("youtube_url", "").strip()
    if not youtube_url:
        return "Please provide a YouTube link."

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(MUSIC_DIR, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            song_title = info.get('title', 'downloaded_song')
        return f"Downloaded and saved in 'music' folder as {song_title}.mp3"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
