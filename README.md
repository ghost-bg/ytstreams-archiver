# YouTube Stream VOD Archiver (yt-dlp Web UI)

A focused web app for archiving YouTube livestream VODs with `yt-dlp`, while still supporting normal YouTube videos.

## Features
- Archive YouTube livestream VODs and standard videos from a single workflow
- Fetch formats available for a specific URL
- Pick an exact `format_id` (or use auto best)
- Customize the output file name
- Download in browser, or upload directly to Google Drive
- Browse Google Drive folders with nested navigation and select by name
- Queue multiple archive jobs and process them sequentially
- Batch queueing from a list of URLs using the same preset
- See estimated completion time for the active job and queue
- Optional low-storage mode for browser downloads (auto-cleans temp file after send)
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

## Notes
- Files are saved in `downloads/`.
- If no format is selected, yt-dlp default format selection is used.
- The app is optimized for YouTube stream VOD archiving, but normal videos are fully supported.
- Google Drive destination uses an optimized spool upload (memory-first, then temp spill for large files), then uploads in chunks.
- Google upload uses Drive scope `drive.file`.
- Folder browsing uses Drive metadata scope `drive.metadata.readonly`.
- In-app OAuth settings are stored in `google_oauth_settings.json`.
- Custom filename changes the base name; the file extension is kept automatically.
- Drive uploads show an in-page progress indicator while processing.
- If you get `redirect_uri_mismatch`, add the exact URI shown in the app's Google OAuth Settings.
- `http://localhost:5000/google/callback` and `http://127.0.0.1:5000/google/callback` are different to Google; add whichever you use (or both).
- Optional: set `GOOGLE_REDIRECT_URI` to force a specific callback URL.
- If folder list is incomplete after this update, reconnect Google Drive to grant the new scope.
- For long videos, increase download timeout via `YTDLP_DOWNLOAD_TIMEOUT_SECONDS` (default: `7200`).
- Transcription intermediates are created in a temp work dir (`YTDLP_TRANSCRIBE_TMP_DIR`, default: system temp dir) and cleaned automatically.
- UI shows free space for download/transcription work dirs and warns when queued jobs are likely to exceed remaining local space.
- Batch queue resolution preset can target common heights (2160p/1440p/1080p/720p/480p/420p/etc.) with automatic fallback to next best available.
- Batch mode can auto-name files from video titles and optionally drop to lower resolutions when estimated size exceeds a per-video MB cap.
- Cancel the current active download from the progress panel.
- Transcription options:
  - Enable "Transcribe and attach subtitle track" to run local `faster-whisper` and attach subtitles as an MKV subtitle track.
  - Enable "Download live chat replay and attach danmaku subtitle track" to add an optional chat overlay subtitle track.
  - When both transcript and chat are enabled, output MKV includes three subtitle tracks:
    - transcript only
    - chat danmaku only
    - merged transcript + chat
  - Subtitle language supports auto-detect or a selected language.
  - Whisper model can be selected per job (`tiny` -> `large-v3`) for speed vs. quality.
  - Transcription quality preset controls decoding aggressiveness (`Fast`, `Balanced`, `High accuracy`).
  - Whisper model settings can be tuned with env vars: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`.
