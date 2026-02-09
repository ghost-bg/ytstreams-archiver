# YouTube Stream VOD Archiver (yt-dlp Web UI)

A web app for archiving YouTube livestream VODs with `yt-dlp`.

## Features
- Upload directly to Google Drive
- Queue multiple jobs
- Batch queueing from a list of URLs using the same preset
- Optional local transcription with `faster-whisper` and subtitle-track attachment (MKV output)
- Optional YouTube live-chat replay extraction and danmaku subtitle-track attachment (MKV output)

## Requirements
- Python 3.10+
- `yt-dlp` installed and available in PATH
- `ffmpeg` installed and available in PATH (required for subtitle-track muxing)

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional: cp .env.example .env and fill values
python app.py
```

Open: http://127.0.0.1:5000

## Google Drive setup
1. In Google Cloud Console, create a project (or pick one).
2. Enable the **Google Drive API**.
3. Configure OAuth consent screen.
4. Create OAuth Client ID credentials of type **Web application**.
5. Add this redirect URI:
   - `http://127.0.0.1:5000/google/callback`
6. Provide credentials in one of these ways:
   - In app: paste Client ID + Client Secret in **Google OAuth Settings** and save.
   - Env vars: set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.
   - Fallback: save Google client JSON as `./google_client_secret.json`.
7. Open the app and click **Connect now**.

