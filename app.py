from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from urllib import parse as urlparse_mod
from urllib import request as urlrequest
from urllib.parse import urlparse

from flask import Flask, abort, after_this_request, jsonify, redirect, render_template, request, send_file, session, url_for

try:
    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    from googleapiclient.http import MediaFileUpload
except ImportError:
    GoogleRequest = None
    Credentials = None
    Flow = None
    build = None
    MediaIoBaseUpload = None
    MediaFileUpload = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)
TRANSCRIPTION_WORK_DIR = Path(
    os.environ.get("YTDLP_TRANSCRIBE_TMP_DIR", tempfile.gettempdir())
).expanduser()
TRANSCRIPTION_WORK_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("YTDLP_DOWNLOAD_TIMEOUT_SECONDS", "7200"))
DEFAULT_FORMATS_TIMEOUT_SECONDS = int(os.environ.get("YTDLP_FORMATS_TIMEOUT_SECONDS", "900"))
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]
GOOGLE_CLIENT_SECRET_FILE = BASE_DIR / "google_client_secret.json"
GOOGLE_TOKEN_FILE = BASE_DIR / "google_token.json"
GOOGLE_OAUTH_SETTINGS_FILE = BASE_DIR / "google_oauth_settings.json"
DEFAULT_YOUTUBE_EXTRACTOR_ARGS = os.environ.get("YTDLP_YOUTUBE_EXTRACTOR_ARGS", "")
SAFE_AUTO_FORMAT_SELECTOR = "bestvideo*[protocol!*=m3u8]+bestaudio[protocol!*=m3u8]/best[protocol!*=m3u8]/best"
COMMON_RESOLUTION_OPTIONS = [
    {"code": "auto", "label": "Auto best"},
    {"code": "2160", "label": "2160p (4K)"},
    {"code": "1440", "label": "1440p"},
    {"code": "1080", "label": "1080p"},
    {"code": "720", "label": "720p"},
    {"code": "480", "label": "480p"},
    {"code": "420", "label": "420p"},
    {"code": "360", "label": "360p"},
    {"code": "240", "label": "240p"},
    {"code": "144", "label": "144p"},
]
FORMAT_DISCOVERY_STRATEGIES: list[str | None] = [
    None,
    "youtube:player_client=tv,android,web",
    "youtube:player_client=ios,android,web",
    "youtube:player_client=tv,ios,android,web",
    "youtube:player_client=web",
    "youtube:player_client=default,android_vr;formats=missing_pot",
    "youtube:player_client=android_vr;formats=missing_pot",
]
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_MODEL_OPTIONS = [
    {"code": "tiny", "label": "Tiny (fastest, lowest accuracy)"},
    {"code": "base", "label": "Base (balanced default)"},
    {"code": "small", "label": "Small (better accuracy)"},
    {"code": "medium", "label": "Medium (high accuracy)"},
    {"code": "large-v3", "label": "Large v3 (best accuracy, slowest)"},
]
WHISPER_QUALITY_OPTIONS = [
    {"code": "fast", "label": "Fast"},
    {"code": "balanced", "label": "Balanced"},
    {"code": "high", "label": "High accuracy"},
]
WHISPER_QUALITY_PRESETS = {
    "fast": {"beam_size": 1, "best_of": 1, "temperature": 0.2, "condition_on_previous_text": False},
    "balanced": {"beam_size": 5, "best_of": 5, "temperature": 0.0, "condition_on_previous_text": True},
    "high": {"beam_size": 8, "best_of": 8, "temperature": 0.0, "condition_on_previous_text": True},
}
WHISPER_SUPPORTED_LANGUAGES = [
    {"code": "auto", "label": "Auto detect"},
    {"code": "en", "label": "English"},
    {"code": "es", "label": "Spanish"},
    {"code": "fr", "label": "French"},
    {"code": "de", "label": "German"},
    {"code": "it", "label": "Italian"},
    {"code": "pt", "label": "Portuguese"},
    {"code": "ja", "label": "Japanese"},
    {"code": "ko", "label": "Korean"},
    {"code": "zh", "label": "Chinese"},
    {"code": "ru", "label": "Russian"},
    {"code": "hi", "label": "Hindi"},
]
_WHISPER_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}
_CANCELLED_JOB_IDS: set[str] = set()
_ACTIVE_JOB_PROCESSES: dict[str, set[subprocess.Popen]] = {}
_JOB_LOCK = threading.Lock()


class JobCancelledError(RuntimeError):
    pass


def _oauth_redirect_uri() -> str:
    return (os.environ.get("GOOGLE_REDIRECT_URI") or "").strip()


def _configure_oauth_transport_for_request(redirect_uri: str) -> None:
    parsed = urlparse(redirect_uri)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and host in {"127.0.0.1", "localhost"}:
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    else:
        os.environ.pop("OAUTHLIB_INSECURE_TRANSPORT", None)


def _load_google_oauth_settings() -> tuple[str | None, str | None, str | None]:
    env_client_id = os.environ.get("GOOGLE_CLIENT_ID")
    env_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if env_client_id and env_client_secret:
        return env_client_id.strip(), env_client_secret.strip(), "environment variables"

    if GOOGLE_OAUTH_SETTINGS_FILE.exists():
        try:
            payload = json.loads(GOOGLE_OAUTH_SETTINGS_FILE.read_text())
            client_id = (payload.get("client_id") or "").strip()
            client_secret = (payload.get("client_secret") or "").strip()
            if client_id and client_secret:
                return client_id, client_secret, "in-app settings"
        except Exception:
            pass
    return None, None, None


def _save_google_oauth_settings(client_id: str, client_secret: str) -> None:
    GOOGLE_OAUTH_SETTINGS_FILE.write_text(
        json.dumps({"client_id": client_id, "client_secret": client_secret}, indent=2)
    )


def _clear_google_oauth_settings() -> None:
    if GOOGLE_OAUTH_SETTINGS_FILE.exists():
        GOOGLE_OAUTH_SETTINGS_FILE.unlink()


def _google_client_config(client_id: str, client_secret: str, redirect_uri: str) -> dict:
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": [redirect_uri],
        }
    }


def _google_supported() -> bool:
    return all((GoogleRequest, Credentials, Flow, build, MediaFileUpload, MediaIoBaseUpload))


def _google_setup_error() -> str:
    if not _google_supported():
        return "Google Drive dependencies are missing. Install requirements.txt."
    client_id, client_secret, _ = _load_google_oauth_settings()
    if client_id and client_secret:
        return ""
    if not GOOGLE_CLIENT_SECRET_FILE.exists():
        return (
            "Missing Google OAuth configuration. Add GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET, "
            "use in-app settings, or provide google_client_secret.json."
        )
    return ""


def _load_google_credentials():
    if not _google_supported():
        return None
    if not GOOGLE_TOKEN_FILE.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(GOOGLE_TOKEN_FILE), GOOGLE_SCOPES)
    except Exception:
        return None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleRequest())
            GOOGLE_TOKEN_FILE.write_text(creds.to_json())
        except Exception:
            return None
    return creds if creds and creds.valid else None


def _save_google_credentials(creds) -> None:
    GOOGLE_TOKEN_FILE.write_text(creds.to_json())


def _disconnect_google() -> None:
    creds = _load_google_credentials()
    if creds and getattr(creds, "token", None):
        try:
            data = urlparse_mod.urlencode({"token": creds.token}).encode("utf-8")
            req = urlrequest.Request("https://oauth2.googleapis.com/revoke", data=data, method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urlrequest.urlopen(req, timeout=10):
                pass
        except Exception:
            pass
    if GOOGLE_TOKEN_FILE.exists():
        GOOGLE_TOKEN_FILE.unlink()
    session.pop("oauth_state", None)


def _drive_connected() -> bool:
    return _load_google_credentials() is not None


def _build_drive_service():
    creds = _load_google_credentials()
    if creds is None:
        raise RuntimeError("Google Drive is not connected yet.")
    return build("drive", "v3", credentials=creds)


def _upload_to_drive(file_path: Path, upload_name: str, folder_id: str | None = None) -> tuple[str, str | None]:
    service = _build_drive_service()
    metadata = {"name": upload_name}
    if folder_id:
        metadata["parents"] = [folder_id]
    media = MediaFileUpload(str(file_path), resumable=True)
    result = service.files().create(body=metadata, media_body=media, fields="id,webViewLink").execute()
    return result["id"], result.get("webViewLink")


def _upload_stream_to_drive(
    url: str,
    selected_format: str,
    upload_name: str,
    folder_id: str | None = None,
    extractor_args_override: str | None = None,
    cookies_from_browser: str = "",
    job_id: str = "",
) -> tuple[str, str | None]:
    service = _build_drive_service()
    metadata = {"name": upload_name}
    if folder_id:
        metadata["parents"] = [folder_id]

    cmd = _yt_dlp_cmd_base(
        extractor_args_override=extractor_args_override,
        cookies_from_browser=cookies_from_browser,
    )
    if selected_format:
        cmd.extend(["-f", selected_format])
    cmd.extend(["--no-warnings", "-o", "-", url])

    # Drive resumable uploads require a seekable stream; spool in memory first and
    # spill to temp disk only for large files to keep local storage usage minimal.
    with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as spool:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("yt-dlp is not installed or not available in PATH.") from exc
        _register_job_process(job_id, proc)

        if proc.stdout is None or proc.stderr is None:
            proc.kill()
            _unregister_job_process(job_id, proc)
            raise RuntimeError("Could not start streaming download.")

        try:
            start = time.monotonic()
            while True:
                if _is_job_cancelled(job_id):
                    proc.kill()
                    raise JobCancelledError("Download cancelled.")
                if (time.monotonic() - start) > DEFAULT_DOWNLOAD_TIMEOUT_SECONDS:
                    proc.kill()
                    raise RuntimeError("yt-dlp timed out while downloading.")
                chunk = proc.stdout.read(1024 * 1024)
                if not chunk:
                    break
                spool.write(chunk)
            stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip()
            try:
                return_code = proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise RuntimeError("yt-dlp timed out while downloading.")
        finally:
            _unregister_job_process(job_id, proc)

        if return_code != 0:
            raise RuntimeError(stderr_text or "Download failed.")
        if spool.tell() == 0:
            raise RuntimeError("The downloaded file is empty.")

        spool.seek(0)
        media = MediaIoBaseUpload(
            spool,
            mimetype="application/octet-stream",
            chunksize=8 * 1024 * 1024,
            resumable=True,
        )
        upload_req = service.files().create(body=metadata, media_body=media, fields="id,webViewLink")

        response = None
        try:
            while response is None:
                _, response = upload_req.next_chunk(num_retries=0)
        except Exception as exc:
            raise RuntimeError(f"Drive upload failed: {exc}") from exc

        if not response:
            raise RuntimeError("Drive upload failed: no response from Google Drive.")
        return response["id"], response.get("webViewLink")


def _list_drive_child_folders(parent_id: str | None) -> list[dict]:
    service = _build_drive_service()
    parent = parent_id or "root"
    folders: list[dict] = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=(
                    "mimeType='application/vnd.google-apps.folder' and "
                    f"'{parent}' in parents and trashed=false"
                ),
                spaces="drive",
                corpora="allDrives",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                fields="nextPageToken, files(id,name)",
                orderBy="name_natural",
                pageSize=200,
                pageToken=page_token,
            )
            .execute()
        )
        for folder in response.get("files", []):
            folder_id = folder.get("id")
            name = folder.get("name") or "(unnamed folder)"
            if folder_id:
                folders.append({"id": folder_id, "name": name})
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return folders


def _dir_storage_snapshot(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _mark_job_cancelled(job_id: str) -> None:
    if not job_id:
        return
    with _JOB_LOCK:
        _CANCELLED_JOB_IDS.add(job_id)
        active = list(_ACTIVE_JOB_PROCESSES.get(job_id, set()))
    for proc in active:
        try:
            proc.terminate()
        except Exception:
            pass


def _clear_job_cancelled(job_id: str) -> None:
    if not job_id:
        return
    with _JOB_LOCK:
        _CANCELLED_JOB_IDS.discard(job_id)


def _is_job_cancelled(job_id: str) -> bool:
    if not job_id:
        return False
    with _JOB_LOCK:
        return job_id in _CANCELLED_JOB_IDS


def _register_job_process(job_id: str, proc: subprocess.Popen) -> None:
    if not job_id:
        return
    with _JOB_LOCK:
        _ACTIVE_JOB_PROCESSES.setdefault(job_id, set()).add(proc)


def _unregister_job_process(job_id: str, proc: subprocess.Popen) -> None:
    if not job_id:
        return
    with _JOB_LOCK:
        bucket = _ACTIVE_JOB_PROCESSES.get(job_id)
        if not bucket:
            return
        bucket.discard(proc)
        if not bucket:
            _ACTIVE_JOB_PROCESSES.pop(job_id, None)
            _CANCELLED_JOB_IDS.discard(job_id)


def _run_checked_process(
    cmd: list[str],
    *,
    timeout_seconds: int,
    job_id: str = "",
    missing_error: str = "Required command is not installed or not available in PATH.",
) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(missing_error) from exc

    _register_job_process(job_id, proc)
    start = time.monotonic()
    try:
        while True:
            try:
                stdout_text, stderr_text = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pass
            if _is_job_cancelled(job_id):
                try:
                    proc.terminate()
                except Exception:
                    pass
                raise JobCancelledError("Download cancelled.")
            if (time.monotonic() - start) > timeout_seconds:
                try:
                    proc.terminate()
                except Exception:
                    pass
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)
        result = subprocess.CompletedProcess(cmd, proc.returncode, stdout_text, stderr_text)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=result.returncode,
                cmd=cmd,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result
    finally:
        _unregister_job_process(job_id, proc)


def _sanitize_filename_stem(name: str) -> str:
    cleaned = unicodedata.normalize("NFKC", name or "")
    cleaned = cleaned.replace("\x00", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", cleaned)
    cleaned = cleaned.strip(". ")
    if not cleaned:
        return ""

    # Avoid Windows reserved filenames when users sync Drive files locally.
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    if cleaned.upper() in reserved:
        cleaned = f"file_{cleaned.lower()}"

    # Keep room for extension to avoid filesystem/path errors.
    return cleaned[:180]


def _probe_output_extension(url: str, selected_format: str, cookies_from_browser: str = "") -> str:
    cmd = _yt_dlp_cmd_base(cookies_from_browser=cookies_from_browser) + ["--skip-download", "--print", "ext"]
    if selected_format:
        cmd.extend(["-f", selected_format])
    cmd.append(url)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        for line in (result.stdout or "").splitlines():
            ext = line.strip().lstrip(".")
            if ext:
                return ext
    except Exception:
        pass
    return "bin"


def _probe_output_extension_with_strategy(
    url: str,
    selected_format: str,
    extractor_args_override: str | None,
    cookies_from_browser: str = "",
) -> str:
    cmd = _yt_dlp_cmd_base(
        extractor_args_override=extractor_args_override,
        cookies_from_browser=cookies_from_browser,
    ) + ["--skip-download", "--print", "ext"]
    if selected_format:
        cmd.extend(["-f", selected_format])
    cmd.append(url)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=max(DEFAULT_TIMEOUT_SECONDS, DEFAULT_FORMATS_TIMEOUT_SECONDS),
        )
        for line in (result.stdout or "").splitlines():
            ext = line.strip().lstrip(".")
            if ext:
                return ext
    except Exception:
        pass
    return "bin"


def _build_final_filename(custom_filename: str, extension: str) -> str:
    stem = _sanitize_filename_stem(custom_filename)
    ext = (extension or "bin").lstrip(".")
    if not stem:
        return f"download.{ext}"
    if stem.lower().endswith(f".{ext.lower()}"):
        return stem
    return f"{stem}.{ext}"


def _get_whisper_model(model_name: str):
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed. Install requirements.txt to enable transcription.")
    model_key = (model_name, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
    if model_key not in _WHISPER_MODEL_CACHE:
        try:
            _WHISPER_MODEL_CACHE[model_key] = WhisperModel(
                model_name,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load Whisper model '{model_name}': {exc}") from exc
    return _WHISPER_MODEL_CACHE[model_key]


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _transcribe_to_srt(
    source_file: Path,
    srt_file: Path,
    language_code: str | None,
    model_name: str,
    quality: str,
) -> str:
    model = _get_whisper_model(model_name)
    decode_options = WHISPER_QUALITY_PRESETS.get(quality, WHISPER_QUALITY_PRESETS["balanced"])
    try:
        segments, info = model.transcribe(
            str(source_file),
            language=language_code,
            vad_filter=True,
            **decode_options,
        )
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc

    written = 0
    with srt_file.open("w", encoding="utf-8") as handle:
        for idx, segment in enumerate(segments, start=1):
            text = (segment.text or "").strip()
            if not text:
                continue
            start = _format_srt_timestamp(segment.start)
            end = _format_srt_timestamp(segment.end)
            handle.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
            written += 1

    if written == 0:
        raise RuntimeError("Transcription finished but produced no subtitle segments.")

    detected = (getattr(info, "language", "") or "").strip().lower()
    return detected or (language_code or "und")


def _ass_escape(text: str) -> str:
    value = (text or "").replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", r"\N")
    return value


def _ass_timestamp(seconds: float) -> str:
    total_cs = max(0, int(round(float(seconds) * 100.0)))
    hours = total_cs // 360000
    minutes = (total_cs % 360000) // 6000
    secs = (total_cs % 6000) // 100
    centis = total_cs % 100
    return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"


def _parse_srt_entries(srt_path: Path) -> list[dict]:
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    entries: list[dict] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line = lines[1] if "-->" in lines[1] else lines[0]
        if "-->" not in time_line:
            continue
        start_text, end_text = [part.strip() for part in time_line.split("-->", 1)]

        def _to_seconds(value: str) -> float:
            hh, mm, ss_ms = value.split(":")
            ss, ms = ss_ms.split(",")
            return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

        try:
            start_s = _to_seconds(start_text)
            end_s = _to_seconds(end_text)
        except Exception:
            continue
        content_lines = lines[2:] if time_line == lines[1] else lines[1:]
        text_value = "\n".join(content_lines).strip()
        if not text_value:
            continue
        entries.append({"start": start_s, "end": max(end_s, start_s + 0.2), "text": text_value})
    return entries


def _write_transcript_ass(entries: list[dict], ass_path: Path) -> None:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Transcript,Arial,44,&H00FFFFFF,&H00FFFFFF,&H00302020,&H66000000,0,0,0,0,100,100,0,0,1,2,0,2,80,80,24,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for entry in entries:
        lines.append(
            "Dialogue: 5,"
            f"{_ass_timestamp(entry['start'])},"
            f"{_ass_timestamp(entry['end'])},"
            "Transcript,,0,0,0,,"
            f"{_ass_escape(entry['text'])}"
        )
    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_live_chat_messages(json_path: Path) -> list[dict]:
    text = json_path.read_text(encoding="utf-8", errors="replace")

    def _parse_json_documents(raw: str) -> list[dict | list]:
        payloads: list[dict | list] = []
        stripped = raw.strip()
        if not stripped:
            return payloads

        try:
            full = json.loads(stripped)
            if isinstance(full, (dict, list)):
                return [full]
        except Exception:
            pass

        decoder = json.JSONDecoder()
        idx = 0
        length = len(raw)
        while idx < length:
            while idx < length and raw[idx].isspace():
                idx += 1
            if idx >= length:
                break
            try:
                obj, next_idx = decoder.raw_decode(raw, idx)
                if isinstance(obj, (dict, list)):
                    payloads.append(obj)
                idx = next_idx
            except Exception:
                break
        if payloads:
            return payloads

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, (dict, list)):
                    payloads.append(obj)
            except Exception:
                continue
        return payloads

    def _iter_dicts(value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from _iter_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from _iter_dicts(child)

    def _extract_runs_text(message_obj: dict) -> str:
        if not isinstance(message_obj, dict):
            return ""
        if isinstance(message_obj.get("simpleText"), str):
            return message_obj["simpleText"].strip()
        runs = message_obj.get("runs")
        if not isinstance(runs, list):
            return ""
        parts: list[str] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            txt = run.get("text")
            if isinstance(txt, str):
                parts.append(txt)
                continue
            emoji = run.get("emoji")
            if isinstance(emoji, dict):
                shortcuts = emoji.get("shortcuts")
                if isinstance(shortcuts, list) and shortcuts and isinstance(shortcuts[0], str):
                    parts.append(shortcuts[0])
        return "".join(parts).strip()

    def _first_numeric_for_keys(value, keys: set[str]):
        for obj in _iter_dicts(value):
            for key in keys:
                if key in obj:
                    try:
                        return float(str(obj[key]))
                    except Exception:
                        continue
        return None

    def _extract_text_from_chat_payload(value) -> str:
        for obj in _iter_dicts(value):
            if "message" in obj and isinstance(obj["message"], dict):
                text_value = _extract_runs_text(obj["message"])
                if text_value:
                    return text_value
            if "headerSubtext" in obj and isinstance(obj["headerSubtext"], dict):
                text_value = _extract_runs_text(obj["headerSubtext"])
                if text_value:
                    return text_value
            if "purchaseAmountText" in obj and isinstance(obj["purchaseAmountText"], dict):
                text_value = _extract_runs_text(obj["purchaseAmountText"])
                if text_value:
                    return text_value
        return ""

    payloads = _parse_json_documents(text)
    if not payloads:
        raise RuntimeError("Could not parse live chat replay JSON: no valid JSON documents found.")

    messages: list[dict] = []
    for payload in payloads:
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            for event in payload.get("events", []):
                segs = event.get("segs") or []
                text_value = "".join((seg.get("utf8") or "") for seg in segs if isinstance(seg, dict)).strip()
                if not text_value:
                    continue
                try:
                    start_ms = int(event.get("tStartMs") or 0)
                except (TypeError, ValueError):
                    continue
                duration_ms = event.get("dDurationMs")
                try:
                    duration_ms_int = int(duration_ms) if duration_ms is not None else 0
                except (TypeError, ValueError):
                    duration_ms_int = 0
                messages.append(
                    {
                        "start": max(0.0, start_ms / 1000.0),
                        "end": max(0.1, (start_ms + max(duration_ms_int, 0)) / 1000.0),
                        "text": text_value,
                    }
                )
            continue

        replay_actions = []
        if isinstance(payload, dict):
            if isinstance(payload.get("actions"), list):
                replay_actions.extend(payload.get("actions") or [])
            if isinstance(payload.get("replayChatItemAction"), dict):
                replay_actions.append({"replayChatItemAction": payload["replayChatItemAction"]})
            if isinstance(payload.get("addChatItemAction"), dict):
                replay_actions.append({"addChatItemAction": payload["addChatItemAction"]})

        for action in replay_actions:
            root = action.get("replayChatItemAction") if isinstance(action, dict) else None
            if isinstance(root, dict):
                nested_actions = root.get("actions")
                if isinstance(nested_actions, list):
                    for nested in nested_actions:
                        text_value = _extract_text_from_chat_payload(nested)
                        if not text_value:
                            continue
                        offset_ms = _first_numeric_for_keys(
                            root,
                            {"videoOffsetTimeMsec", "videoOffsetTimeMs", "timestampUsec"},
                        )
                        if offset_ms is None:
                            offset_ms = _first_numeric_for_keys(
                                nested,
                                {"videoOffsetTimeMsec", "videoOffsetTimeMs", "timestampUsec"},
                            )
                        if offset_ms is None:
                            offset_ms = 0.0
                        if offset_ms > 1_000_000_000:  # likely microseconds
                            offset_ms = offset_ms / 1000.0
                        start_s = max(0.0, float(offset_ms) / 1000.0)
                        messages.append({"start": start_s, "end": start_s + 6.0, "text": text_value})
                continue

            text_value = _extract_text_from_chat_payload(action)
            if not text_value:
                continue
            offset_ms = _first_numeric_for_keys(
                action,
                {"videoOffsetTimeMsec", "videoOffsetTimeMs", "timestampUsec"},
            )
            if offset_ms is None:
                offset_ms = 0.0
            if offset_ms > 1_000_000_000:
                offset_ms = offset_ms / 1000.0
            start_s = max(0.0, float(offset_ms) / 1000.0)
            messages.append({"start": start_s, "end": start_s + 6.0, "text": text_value})

    messages.sort(key=lambda x: float(x.get("start") or 0.0))
    if not messages:
        raise RuntimeError("Live chat replay exists but no chat message events were extracted.")
    return messages


def _write_danmaku_ass(chat_messages: list[dict], ass_path: Path, transcript_entries: list[dict] | None = None) -> None:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Danmaku,Arial,40,&H00FFFFFF,&H00FFFFFF,&H00303030,&H66000000,0,0,0,0,100,100,0,0,1,1.5,0,7,24,24,24,1",
        "Style: Transcript,Arial,44,&H00FFFFFF,&H00FFFFFF,&H00302020,&H66000000,0,0,0,0,100,100,0,0,1,2,0,2,80,80,24,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lane_count = 12
    lane_height = 56
    lane_start_y = 96
    lane_available_at = [0.0] * lane_count
    for msg in sorted(chat_messages, key=lambda x: float(x.get("start") or 0.0)):
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        start_s = float(msg.get("start") or 0.0)
        visible_seconds = min(14.0, max(5.0, 4.0 + len(text) / 7.0))
        end_s = max(float(msg.get("end") or 0.0), start_s + visible_seconds)
        lane_idx = min(range(lane_count), key=lambda i: lane_available_at[i])
        if lane_available_at[lane_idx] > start_s:
            start_s = lane_available_at[lane_idx] + 0.05
            end_s = start_s + visible_seconds
        lane_available_at[lane_idx] = start_s + visible_seconds * 0.6
        y = lane_start_y + lane_idx * lane_height
        text_width = max(120, min(1700, int(len(text) * 22)))
        x_from = 1980
        x_to = -text_width
        override = f"{{\\move({x_from},{y},{x_to},{y})\\alpha&H55&}}"
        lines.append(
            "Dialogue: 4,"
            f"{_ass_timestamp(start_s)},"
            f"{_ass_timestamp(end_s)},"
            "Danmaku,,0,0,0,,"
            f"{override}{_ass_escape(text)}"
        )

    for entry in transcript_entries or []:
        lines.append(
            "Dialogue: 6,"
            f"{_ass_timestamp(entry['start'])},"
            f"{_ass_timestamp(entry['end'])},"
            "Transcript,,0,0,0,,"
            f"{_ass_escape(entry['text'])}"
        )

    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _download_live_chat_json3(
    *,
    url: str,
    request_id: str,
    extractor_args_override: str | None,
    cookies_from_browser: str,
    output_dir: Path,
    job_id: str = "",
) -> Path:
    output_template = str(output_dir / f"{request_id}.%(ext)s")
    cmd = _yt_dlp_cmd_base(
        extractor_args_override=extractor_args_override,
        cookies_from_browser=cookies_from_browser,
    ) + [
        "--skip-download",
        "--no-warnings",
        "--write-subs",
        "--sub-langs",
        "live_chat",
        "--sub-format",
        "json3",
        "-o",
        output_template,
        url,
    ]
    try:
        _run_checked_process(
            cmd,
            timeout_seconds=max(DEFAULT_TIMEOUT_SECONDS, DEFAULT_FORMATS_TIMEOUT_SECONDS),
            job_id=job_id,
            missing_error="yt-dlp is not installed or not available in PATH.",
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("yt-dlp timed out while downloading live chat replay.") from exc
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stderr or exc.stdout or "Could not fetch live chat replay.").strip()
        raise RuntimeError(error_output) from exc

    candidates = sorted(
        [
            p
            for p in output_dir.glob(f"{request_id}*")
            if p.is_file()
            and p.suffix.lower() in {".json3", ".json"}
            and ("live_chat" in p.name or "livechat" in p.name)
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            [
                p
                for p in output_dir.glob(f"{request_id}*")
                if p.is_file() and p.suffix.lower() in {".json3", ".json"}
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise RuntimeError("Live chat replay was not found for this VOD (it may be unavailable or disabled).")
    return candidates[0]


def _attach_subtitle_tracks(
    input_file: Path,
    subtitle_tracks: list[dict],
    output_file: Path,
    job_id: str = "",
) -> None:
    if not subtitle_tracks:
        raise RuntimeError("No subtitle tracks were provided for muxing.")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
    ]
    for track in subtitle_tracks:
        cmd.extend(["-i", str(track["path"])])
    cmd.extend(["-map", "0"])
    for idx in range(len(subtitle_tracks)):
        cmd.extend(["-map", f"{idx + 1}:0"])
    cmd.extend(["-c", "copy"])
    for idx, track in enumerate(subtitle_tracks):
        cmd.extend(["-c:s:" + str(idx), str(track.get("codec") or "srt")])
        cmd.extend(["-metadata:s:s:" + str(idx), f"language={track.get('language') or 'und'}"])
        if track.get("title"):
            cmd.extend(["-metadata:s:s:" + str(idx), f"title={track['title']}"])
        cmd.extend(["-disposition:s:" + str(idx), "default" if track.get("default") else "0"])
    cmd.append(str(output_file))
    try:
        _run_checked_process(
            cmd,
            timeout_seconds=max(DEFAULT_TIMEOUT_SECONDS, 600),
            job_id=job_id,
            missing_error="ffmpeg is not installed or not available in PATH.",
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out while attaching subtitles.") from exc
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stderr or exc.stdout or "Failed to attach subtitles.").strip()
        raise RuntimeError(error_output) from exc


def _resolve_downloaded_output_file(request_id: str, result_stdout: str, output_dir: Path) -> Path:
    for line in (result_stdout or "").splitlines():
        line = line.strip()
        if line:
            candidate = Path(line)
            if candidate.exists():
                return candidate

    matches = sorted(
        (
            p for p in output_dir.glob(f"{request_id}*")
            if p.is_file() and not p.name.endswith(".part") and not p.name.endswith(".ytdl")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise RuntimeError("yt-dlp completed but no file was produced.")
    return matches[0]


def _render_index(
    *,
    url: str = "",
    formats: list[dict] | None = None,
    error: str | None = None,
    info: str | None = None,
    video_title: str | None = None,
    drive_link: str | None = None,
    selected_destination: str = "browser",
    selected_drive_folder_id: str = "",
    custom_filename: str = "",
    selected_format_strategy: str = "",
    cookies_from_browser: str = "",
    allow_fallback: str = "0",
    prefer_non_hls_auto: str = "1",
    execution_log: str = "",
    transcribe_subtitles: str = "0",
    download_chat_danmaku: str = "0",
    transcription_language: str = "auto",
    transcription_model: str = "",
    transcription_quality: str = "balanced",
    preferred_resolution: str = "auto",
):
    oauth_client_id, _, oauth_source = _load_google_oauth_settings()
    configured_redirect_uri = _oauth_redirect_uri() or url_for("google_callback", _external=True)
    drive_connected = _drive_connected()
    selected_model = transcription_model or WHISPER_MODEL_NAME
    model_options = list(WHISPER_MODEL_OPTIONS)
    known_models = {item["code"] for item in model_options}
    if selected_model not in known_models:
        model_options.append({"code": selected_model, "label": f"{selected_model} (custom)"})
    return render_template(
        "index.html",
        url=url,
        formats=formats,
        error=error,
        info=info,
        video_title=video_title,
        drive_link=drive_link,
        drive_connected=drive_connected,
        drive_setup_error=_google_setup_error(),
        selected_destination=selected_destination,
        selected_drive_folder_id=selected_drive_folder_id,
        custom_filename=custom_filename,
        selected_format_strategy=selected_format_strategy,
        cookies_from_browser=cookies_from_browser,
        allow_fallback=allow_fallback,
        prefer_non_hls_auto=prefer_non_hls_auto,
        execution_log=execution_log,
        transcribe_subtitles=transcribe_subtitles,
        download_chat_danmaku=download_chat_danmaku,
        transcription_language=transcription_language,
        transcription_model=selected_model,
        transcription_quality=transcription_quality,
        transcription_languages=WHISPER_SUPPORTED_LANGUAGES,
        transcription_models=model_options,
        transcription_quality_options=WHISPER_QUALITY_OPTIONS,
        preferred_resolution=preferred_resolution,
        resolution_options=COMMON_RESOLUTION_OPTIONS,
        oauth_client_id=oauth_client_id or "",
        oauth_source=oauth_source or ("google_client_secret.json" if GOOGLE_CLIENT_SECRET_FILE.exists() else "not configured"),
        oauth_redirect_uri=configured_redirect_uri,
    )


@app.get("/")
def index():
    info = request.args.get("info")
    error = request.args.get("error")
    return _render_index(url="", formats=None, error=error, info=info, video_title=None)


def _human_size(num_bytes: int | None) -> str:
    if not num_bytes:
        return ""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _yt_dlp_cmd_base(
    extractor_args_override: str | None = None,
    cookies_from_browser: str = "",
) -> list[str]:
    cmd = ["yt-dlp", "--no-playlist", "--no-progress"]
    extractor_args = DEFAULT_YOUTUBE_EXTRACTOR_ARGS if extractor_args_override == "__DEFAULT__" else extractor_args_override
    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    return cmd


def _is_retryable_download_error(message: str) -> bool:
    lowered = (message or "").lower()
    retry_markers = (
        "403 forbidden",
        "http error 403",
        "access denied",
        "ffmpeg exited with code 8",
        "downloaded file is empty",
    )
    return any(marker in lowered for marker in retry_markers)


def _selector_for_preferred_resolution(preferred_resolution: str, prefer_non_hls_auto: bool) -> str:
    code = (preferred_resolution or "auto").strip().lower()
    if code == "auto":
        return SAFE_AUTO_FORMAT_SELECTOR if prefer_non_hls_auto else ""
    try:
        height = int(code)
    except ValueError:
        return SAFE_AUTO_FORMAT_SELECTOR if prefer_non_hls_auto else ""
    non_hls = (
        f"bestvideo*[height<={height}][protocol!*=m3u8]+bestaudio[protocol!*=m3u8]/"
        f"best[height<={height}][protocol!*=m3u8]/"
    )
    generic = f"bestvideo*[height<={height}]+bestaudio/best[height<={height}]/best"
    return f"{non_hls}{generic}" if prefer_non_hls_auto else generic


def _choose_batch_format_selector(
    formats: list[dict],
    preferred_resolution: str,
    max_filesize_bytes: int | None,
) -> str:
    video_formats = [item for item in formats if item.get("has_video")]
    if not video_formats:
        return ""

    target_height = None
    if preferred_resolution != "auto":
        try:
            target_height = int(preferred_resolution)
        except ValueError:
            target_height = None

    if target_height:
        constrained = [item for item in video_formats if int(item.get("height") or 0) <= target_height]
        if constrained:
            video_formats = constrained

    if max_filesize_bytes and max_filesize_bytes > 0:
        sized = [
            item
            for item in video_formats
            if isinstance(item.get("size_bytes"), int) and int(item["size_bytes"]) <= max_filesize_bytes
        ]
        if sized:
            video_formats = sized

    chosen = video_formats[0] if video_formats else None
    return (chosen or {}).get("selector") or ""


def _resolve_batch_metadata(
    *,
    url: str,
    preferred_resolution: str,
    max_filesize_bytes: int | None,
    cookies_from_browser: str,
) -> tuple[str, str, str]:
    try:
        video_title, format_rows, format_strategy, _ = _load_formats(url, cookies_from_browser=cookies_from_browser)
    except RuntimeError:
        return "", "", ""

    selector = _choose_batch_format_selector(
        formats=format_rows or [],
        preferred_resolution=preferred_resolution,
        max_filesize_bytes=max_filesize_bytes,
    )
    return video_title or "", selector, format_strategy or ""


def _format_retry_candidates(selected_format: str) -> list[str]:
    # Keep user choice first, then try safer non-HLS and generic fallbacks.
    candidates = [
        selected_format,
        "bestvideo*[protocol!*=m3u8]+bestaudio[protocol!*=m3u8]/best[protocol!*=m3u8]/best",
        "bestvideo+bestaudio/best",
        "",
    ]
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _download_attempts(
    selected_format: str,
    selected_format_strategy: str,
    *,
    allow_fallback: bool,
) -> list[tuple[str | None, str]]:
    if not allow_fallback:
        return [(selected_format_strategy or None, selected_format)]
    format_candidates = _format_retry_candidates(selected_format)
    strategy_candidates = [
        selected_format_strategy or None,
        "youtube:player_client=tv,ios,android,web",
        "youtube:player_client=tv,android,web",
        "youtube:player_client=ios,android,web",
        "youtube:player_client=android,web",
        None,
    ]
    attempts: list[tuple[str | None, str]] = []
    seen: set[tuple[str | None, str]] = set()
    for strategy in strategy_candidates:
        for format_selector in format_candidates:
            key = (strategy, format_selector)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(key)
    return attempts


def _run_download_with_options(
    url: str,
    output_template: str,
    selected_format: str,
    extractor_args_override: str | None,
    cookies_from_browser: str = "",
    job_id: str = "",
):
    cmd = _yt_dlp_cmd_base(
        extractor_args_override=extractor_args_override,
        cookies_from_browser=cookies_from_browser,
    ) + [
        "--print",
        "after_move:filepath",
        "-o",
        output_template,
    ]
    if selected_format:
        cmd.extend(["-f", selected_format])
    cmd.append(url)
    return _run_checked_process(
        cmd,
        timeout_seconds=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        job_id=job_id,
        missing_error="yt-dlp is not installed or not available in PATH.",
    )


def _download_to_local_file(
    url: str,
    request_id: str,
    selected_format: str,
    selected_format_strategy: str,
    allow_fallback: bool,
    cookies_from_browser: str,
    output_dir: Path = DOWNLOAD_DIR,
    job_id: str = "",
) -> Path:
    output_template = str(output_dir / f"{request_id}.%(ext)s")
    result = None
    last_download_error = ""
    try:
        for attempt_strategy, attempt_format in _download_attempts(
            selected_format,
            selected_format_strategy,
            allow_fallback=allow_fallback,
        ):
            try:
                result = _run_download_with_options(
                    url=url,
                    output_template=output_template,
                    selected_format=attempt_format,
                    extractor_args_override=attempt_strategy,
                    cookies_from_browser=cookies_from_browser,
                    job_id=job_id,
                )
                break
            except JobCancelledError:
                raise
            except subprocess.CalledProcessError as exc:
                error_output = (exc.stderr or exc.stdout or "Download failed.").strip()
                last_download_error = error_output
                if _is_retryable_download_error(error_output):
                    continue
                raise RuntimeError(error_output) from exc
    except JobCancelledError:
        raise
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is not installed or not available in PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("yt-dlp timed out while downloading.") from exc

    if result is None:
        raise RuntimeError(last_download_error or "Download failed.")
    return _resolve_downloaded_output_file(request_id, result.stdout or "", output_dir)


def _verify_selected_format_available(
    url: str,
    selected_format: str,
    extractor_args_override: str | None,
    cookies_from_browser: str = "",
) -> tuple[bool, str]:
    if not selected_format:
        return True, ""
    cmd = _yt_dlp_cmd_base(
        extractor_args_override=extractor_args_override,
        cookies_from_browser=cookies_from_browser,
    ) + [
        "--skip-download",
        "--no-warnings",
        "--print",
        "ext",
        "-f",
        selected_format,
        url,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=max(DEFAULT_TIMEOUT_SECONDS, 120),
        )
        return True, ""
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stderr or exc.stdout or "Requested format is not available.").strip()
        return False, error_output
    except subprocess.TimeoutExpired:
        return False, "Timed out while verifying selected format availability."
    except FileNotFoundError:
        return False, "yt-dlp is not installed or not available in PATH."


def _format_size_bytes(item: dict) -> int | None:
    size = item.get("filesize") or item.get("filesize_approx")
    if not size:
        return None
    try:
        return int(size)
    except (TypeError, ValueError):
        return None


def _video_with_audio_selector(format_id: str, is_hls: bool) -> str:
    # Prefer non-HLS audio first when possible; HLS pairings can fail at segment open
    # time for long VODs even when format listing succeeds.
    if is_hls:
        return (
            f"{format_id}+251/"
            f"{format_id}+250/"
            f"{format_id}+249/"
            f"{format_id}+140/"
            f"{format_id}+bestaudio[protocol!*=m3u8]/"
            f"{format_id}+bestaudio/{format_id}"
        )
    return f"{format_id}+bestaudio/{format_id}"


def _extract_formats(video_json: dict, strategy: str | None = None) -> list[dict]:
    best_audio_size = None
    best_audio_bitrate = -1.0
    for item in video_json.get("formats", []):
        vcodec = item.get("vcodec")
        acodec = item.get("acodec")
        if vcodec == "none" and acodec and acodec != "none":
            abr = item.get("abr") or item.get("tbr") or 0
            try:
                abr_value = float(abr)
            except (TypeError, ValueError):
                abr_value = 0.0
            size = _format_size_bytes(item)
            if abr_value > best_audio_bitrate:
                best_audio_bitrate = abr_value
                best_audio_size = size

    formats = []
    for item in video_json.get("formats", []):
        format_id = item.get("format_id")
        if not format_id:
            continue
        vcodec = item.get("vcodec")
        acodec = item.get("acodec")
        if vcodec == "none" and acodec == "none":
            continue
        ext = item.get("ext") or "?"
        protocol = str(item.get("protocol") or "")
        is_hls = "m3u8" in protocol
        resolution = item.get("resolution")
        if not resolution:
            width = item.get("width")
            height = item.get("height")
            if width and height:
                resolution = f"{width}x{height}"
            elif item.get("vcodec") == "none":
                resolution = "audio only"
            else:
                resolution = "unknown"
        filesize = _format_size_bytes(item)
        bitrate = item.get("tbr") or item.get("abr")

        # Make video-only selections playable by auto-merging best available audio.
        if vcodec != "none" and acodec == "none":
            selector = _video_with_audio_selector(format_id, is_hls)
            mode = "video + best audio"
            merged_size = filesize + best_audio_size if filesize and best_audio_size else None
            size_text = _human_size(merged_size)
            quality = f"~{size_text}" if size_text else "size unavailable"
            size_bytes = merged_size
        elif vcodec != "none" and acodec != "none":
            selector = format_id
            mode = "video+audio"
            size_text = _human_size(filesize)
            quality = size_text if size_text else "size unavailable"
            size_bytes = filesize
        elif acodec != "none" and vcodec == "none":
            selector = format_id
            mode = "audio only"
            size_text = _human_size(filesize)
            quality = size_text if size_text else "size unavailable"
            size_bytes = filesize
        else:
            continue

        formats.append(
            {
                "selector": selector,
                "strategy": strategy or "",
                "size_bytes": int(size_bytes) if size_bytes else None,
                "is_hls": is_hls,
                "has_video": vcodec != "none",
                "height": item.get("height") or 0,
                "bitrate": float(bitrate) if bitrate else 0.0,
                "label": (
                    f"{resolution}, {ext}, {mode}, {quality}"
                    f"{', hls stream' if is_hls else ''}"
                    f"{f', {int(float(bitrate))} kbps' if bitrate and vcodec == 'none' else ''}"
                    f" [{format_id}]"
                ),
            }
        )
    return sorted(
        formats,
        key=lambda x: (
            not x["has_video"],
            bool(x.get("is_hls")),
            x.get("size_bytes") is None,
            -int(x["height"]),
            -x["bitrate"],
        ),
    )


def _formats_score(formats: list[dict]) -> tuple[int, int, int]:
    video_items = [f for f in formats if f.get("has_video")]
    max_height = max((int(f.get("height") or 0) for f in video_items), default=0)
    return (max_height, len(video_items), len(formats))


def _run_formats_json(
    url: str,
    extractor_args_override: str | None = None,
    cookies_from_browser: str = "",
) -> dict:
    cmd = _yt_dlp_cmd_base(
        extractor_args_override=extractor_args_override,
        cookies_from_browser=cookies_from_browser,
    ) + ["--no-warnings", "-J", url]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=max(DEFAULT_TIMEOUT_SECONDS, DEFAULT_FORMATS_TIMEOUT_SECONDS),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is not installed or not available in PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("yt-dlp timed out while fetching formats.") from exc
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stderr or exc.stdout or "Failed to fetch formats.").strip()
        # Some yt-dlp/cookie contexts incorrectly fail -J with a format selection
        # error; retry with a plain command and no extractor args.
        if "requested format is not available" in error_output.lower():
            plain_cmd = ["yt-dlp", "--no-playlist", "--no-progress"]
            if cookies_from_browser:
                plain_cmd.extend(["--cookies-from-browser", cookies_from_browser])
            plain_cmd.extend(["--no-warnings", "-J", url])
            try:
                result = subprocess.run(
                    plain_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=max(DEFAULT_TIMEOUT_SECONDS, DEFAULT_FORMATS_TIMEOUT_SECONDS),
                )
            except subprocess.CalledProcessError as retry_exc:
                retry_output = (retry_exc.stderr or retry_exc.stdout or error_output).strip()
                raise RuntimeError(retry_output) from retry_exc
            except subprocess.TimeoutExpired as retry_exc:
                raise RuntimeError("yt-dlp timed out while fetching formats.") from retry_exc
        else:
            raise RuntimeError(error_output) from exc

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("yt-dlp returned invalid format metadata.") from exc


def _run_formats_table(
    url: str,
    extractor_args_override: str | None = None,
    cookies_from_browser: str = "",
) -> str:
    cmd = _yt_dlp_cmd_base(
        extractor_args_override=extractor_args_override,
        cookies_from_browser=cookies_from_browser,
    ) + ["--no-warnings", "-F", url]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=max(DEFAULT_TIMEOUT_SECONDS, DEFAULT_FORMATS_TIMEOUT_SECONDS),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is not installed or not available in PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("yt-dlp timed out while fetching formats.") from exc
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stderr or exc.stdout or "Failed to fetch formats.").strip()
        raise RuntimeError(error_output) from exc
    return result.stdout or ""


def _parse_formats_table(table_text: str, strategy: str | None = None) -> list[dict]:
    formats: list[dict] = []
    in_table = False
    for raw_line in (table_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower_line = line.lower()
        if lower_line.startswith("format code"):
            in_table = True
            continue
        if re.search(r"\bid\b", line, flags=re.IGNORECASE) and re.search(r"\bext\b", line, flags=re.IGNORECASE):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("["):
            continue
        if re.fullmatch(r"[-\s|]+", line):
            continue

        match = re.match(r"^(\S+)\s+(\S+)\s+(.+)$", line)
        if not match:
            continue
        format_id, ext, details = match.groups()
        # Ignore non-format noise and storyboard/image rows.
        if not re.fullmatch(r"[A-Za-z0-9._-]+", format_id):
            continue
        if not re.fullmatch(r"[A-Za-z0-9._-]+", ext):
            continue
        details_lower = details.lower()
        if format_id.lower().startswith("sb") or ext.lower() == "mhtml" or "storyboard" in details_lower:
            continue
        if "images" in details_lower and "video" not in details_lower and "audio" not in details_lower:
            continue

        has_video = "audio only" not in details_lower
        if has_video and "video only" in details_lower:
            selector = _video_with_audio_selector(format_id, is_hls)
            mode = "video + best audio"
        elif has_video:
            selector = format_id
            mode = "video+audio"
        else:
            selector = format_id
            mode = "audio only"

        height = 0
        m_res = re.search(r"(\d{3,4})p", details_lower)
        if m_res:
            height = int(m_res.group(1))
        else:
            m_dims = re.search(r"(\d{3,5})x(\d{3,5})", details_lower)
            if m_dims:
                height = int(m_dims.group(2))

        bitrate = 0.0
        m_kbps = re.search(r"(\d+(?:\.\d+)?)\s*kbps", details_lower)
        if m_kbps:
            bitrate = float(m_kbps.group(1))

        is_hls = "m3u8" in details_lower or "hls" in details_lower
        label = f"{ext}, {mode}, {details} [{format_id}]"
        if is_hls:
            label = f"{label}, hls stream"

        formats.append(
            {
                "selector": selector,
                "strategy": strategy or "",
                "size_bytes": None,
                "is_hls": is_hls,
                "has_video": has_video,
                "height": height,
                "bitrate": bitrate,
                "label": label,
            }
        )
    return sorted(
        formats,
        key=lambda x: (
            not x["has_video"],
            bool(x.get("is_hls")),
            x.get("size_bytes") is None,
            -int(x["height"]),
            -x["bitrate"],
        ),
    )


def _synthetic_fallback_formats(strategy: str | None = None) -> list[dict]:
    return [
        {
            "selector": SAFE_AUTO_FORMAT_SELECTOR,
            "strategy": strategy or "",
            "size_bytes": None,
            "is_hls": False,
            "has_video": True,
            "height": 0,
            "bitrate": 0.0,
            "label": "auto, video + audio, best non-HLS (safe fallback)",
        },
        {
            "selector": "bestvideo+bestaudio/best",
            "strategy": strategy or "",
            "size_bytes": None,
            "is_hls": False,
            "has_video": True,
            "height": 0,
            "bitrate": 0.0,
            "label": "auto, video + audio, generic best (fallback)",
        },
    ]


def _load_formats(url: str, cookies_from_browser: str = "") -> tuple[str | None, list[dict], str, str]:
    # Try multiple client strategies and keep the richest/highest-res format set.
    # This improves long livestream VOD handling where some clients expose only 360p.
    strategies = FORMAT_DISCOVERY_STRATEGIES

    best_payload = None
    best_formats: list[dict] = []
    best_score = (-1, -1, -1)
    best_strategy = ""
    last_error: RuntimeError | None = None
    debug_lines: list[str] = [f"Format discovery for URL: {url}"]

    for strategy in strategies:
        strategy_label = strategy if strategy else "(default)"
        debug_lines.append(f"- Try strategy: {strategy_label}")
        try:
            payload = _run_formats_json(
                url,
                extractor_args_override=strategy,
                cookies_from_browser=cookies_from_browser,
            )
            formats = _extract_formats(payload, strategy)
            score = _formats_score(formats)
            debug_lines.append(
                f"  -> JSON ok: formats={len(formats)}, max_height={score[0]}, "
                f"video_count={score[1]}, total_count={score[2]}"
            )
            if score > best_score:
                best_payload = payload
                best_formats = formats
                best_score = score
                best_strategy = strategy or ""
                debug_lines.append(f"  -> Selected as current best strategy: {strategy_label}")
            # Good enough result; avoid extra expensive probes.
            if best_score[0] >= 1080 and best_score[1] >= 4:
                debug_lines.append("  -> Early stop: sufficient high-res results found")
                break
        except RuntimeError as exc:
            last_error = exc
            debug_lines.append(f"  -> JSON failed: {exc}")

    if best_payload is None:
        # Fallback: parse `-F` table when `-J` JSON extraction is unstable.
        fallback_errors: list[str] = []
        for strategy in strategies:
            try:
                table = _run_formats_table(
                    url,
                    extractor_args_override=strategy,
                    cookies_from_browser=cookies_from_browser,
                )
                parsed_formats = _parse_formats_table(table, strategy)
                if parsed_formats:
                    debug_lines.append(
                        f"  -> Fallback -F parse ok with strategy {strategy or '(default)'}: "
                        f"formats={len(parsed_formats)}"
                    )
                    return None, parsed_formats, strategy or "", "\n".join(debug_lines)
            except RuntimeError as exc:
                fallback_errors.append(str(exc))
                debug_lines.append(f"  -> Fallback -F failed with strategy {strategy or '(default)'}: {exc}")
        if last_error is not None:
            msg = str(last_error).lower()
            if "requested format is not available" in msg:
                debug_lines.append("  -> Using synthetic fallback formats due to unavailable format error")
                return None, _synthetic_fallback_formats(), "", "\n".join(debug_lines)
            if fallback_errors:
                raise RuntimeError(f"{last_error}\n\nFallback list-formats errors: {' | '.join(fallback_errors[:2])}")
            raise last_error
        if fallback_errors:
            if any("requested format is not available" in err.lower() for err in fallback_errors):
                debug_lines.append("  -> Using synthetic fallback formats after list-formats errors")
                return None, _synthetic_fallback_formats(), "", "\n".join(debug_lines)
            raise RuntimeError(fallback_errors[0])
        raise RuntimeError("Failed to fetch formats.")

    debug_lines.append(
        f"Final strategy: {best_strategy or '(default)'} | "
        f"formats={len(best_formats)} | max_height={best_score[0]}"
    )
    return best_payload.get("title"), best_formats, best_strategy, "\n".join(debug_lines)


@app.post("/formats")
def formats():
    url = (request.form.get("url") or "").strip()
    cookies_from_browser = (request.form.get("cookies_from_browser") or "").strip()
    if not url:
        return _render_index(
            url=url,
            formats=None,
            error="A YouTube URL is required.",
            video_title=None,
            cookies_from_browser=cookies_from_browser,
        ), 400

    try:
        video_title, format_rows, format_strategy, execution_log = _load_formats(
            url, cookies_from_browser=cookies_from_browser
        )
    except RuntimeError as exc:
        return _render_index(
            url=url,
            formats=None,
            error=str(exc),
            video_title=None,
            cookies_from_browser=cookies_from_browser,
            execution_log=str(exc),
        ), 400

    if not format_rows:
        return _render_index(
            url=url,
            formats=None,
            error="No downloadable formats were returned for this URL.",
            video_title=video_title,
            cookies_from_browser=cookies_from_browser,
            execution_log=execution_log,
        ), 400

    return _render_index(
        url=url,
        formats=format_rows,
        error=None,
        video_title=video_title,
        custom_filename=video_title or "",
        selected_format_strategy=format_strategy,
        cookies_from_browser=cookies_from_browser,
        execution_log=execution_log,
    )


@app.post("/formats/json")
def formats_json():
    url = (request.form.get("url") or "").strip()
    cookies_from_browser = (request.form.get("cookies_from_browser") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "A YouTube URL is required.", "execution_log": "A YouTube URL is required."}), 400
    try:
        video_title, format_rows, format_strategy, execution_log = _load_formats(
            url, cookies_from_browser=cookies_from_browser
        )
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc), "execution_log": str(exc)}), 400
    if not format_rows:
        return jsonify(
            {
                "ok": False,
                "error": "No downloadable formats were returned for this URL.",
                "execution_log": execution_log,
            }
        ), 400
    return jsonify(
        {
            "ok": True,
            "url": url,
            "video_title": video_title or "",
            "formats": format_rows,
            "format_strategy": format_strategy or "",
            "execution_log": execution_log,
        }
    )


@app.get("/google/connect")
def google_connect():
    setup_error = _google_setup_error()
    if setup_error:
        return _render_index(error=setup_error), 400

    redirect_uri = _oauth_redirect_uri() or url_for("google_callback", _external=True)
    _configure_oauth_transport_for_request(redirect_uri)
    client_id, client_secret, _ = _load_google_oauth_settings()
    if client_id and client_secret:
        flow = Flow.from_client_config(
            _google_client_config(client_id, client_secret, redirect_uri),
            scopes=GOOGLE_SCOPES,
        )
    else:
        flow = Flow.from_client_secrets_file(str(GOOGLE_CLIENT_SECRET_FILE), scopes=GOOGLE_SCOPES)
    flow.redirect_uri = redirect_uri
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    session["oauth_state"] = state
    return redirect(authorization_url)


@app.get("/google/callback")
def google_callback():
    setup_error = _google_setup_error()
    if setup_error:
        return _render_index(error=setup_error), 400
    state = session.get("oauth_state")
    if not state:
        return _render_index(error="OAuth state missing. Try connecting again."), 400

    try:
        redirect_uri = _oauth_redirect_uri() or url_for("google_callback", _external=True)
        _configure_oauth_transport_for_request(redirect_uri)
        client_id, client_secret, _ = _load_google_oauth_settings()
        if client_id and client_secret:
            flow = Flow.from_client_config(
                _google_client_config(client_id, client_secret, redirect_uri),
                scopes=GOOGLE_SCOPES,
                state=state,
            )
        else:
            flow = Flow.from_client_secrets_file(
                str(GOOGLE_CLIENT_SECRET_FILE),
                scopes=GOOGLE_SCOPES,
                state=state,
            )
        flow.redirect_uri = redirect_uri
        flow.fetch_token(authorization_response=request.url)
        _save_google_credentials(flow.credentials)
    except Exception as exc:
        return _render_index(error=f"Google OAuth failed: {exc}"), 400

    return redirect(url_for("index", info="Google Drive connected."))


@app.post("/settings/google")
def settings_google():
    client_id = (request.form.get("google_client_id") or "").strip()
    client_secret = (request.form.get("google_client_secret") or "").strip()
    action = (request.form.get("settings_action") or "save").strip()

    if action == "clear":
        _clear_google_oauth_settings()
        return redirect(url_for("index", info="Cleared in-app Google OAuth settings."))

    if not client_id or not client_secret:
        return _render_index(error="Both Google Client ID and Client Secret are required to save settings."), 400

    _save_google_oauth_settings(client_id, client_secret)
    return redirect(url_for("index", info="Saved Google OAuth settings."))


@app.post("/google/disconnect")
def google_disconnect():
    _disconnect_google()
    return redirect(url_for("index", info="Disconnected Google Drive."))


@app.get("/google/folders")
def google_folders():
    if not _drive_connected():
        return jsonify({"ok": False, "error": "Google Drive is not connected."}), 401

    parent_id = (request.args.get("parent_id") or "").strip() or None
    try:
        folders = _list_drive_child_folders(parent_id)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Could not load folders: {exc}"}), 400
    return jsonify({"ok": True, "folders": folders})


@app.get("/system/storage")
def system_storage():
    try:
        return jsonify(
            {
                "ok": True,
                "download_dir": _dir_storage_snapshot(DOWNLOAD_DIR),
                "transcribe_tmp_dir": _dir_storage_snapshot(TRANSCRIPTION_WORK_DIR),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Could not read storage usage: {exc}"}), 500


@app.post("/download/cancel")
def cancel_download():
    job_id = (request.form.get("job_id") or "").strip()
    if not job_id:
        return jsonify({"ok": False, "error": "job_id is required."}), 400
    _mark_job_cancelled(job_id)
    return jsonify({"ok": True, "info": "Cancellation requested."})


@app.post("/download")
def download():
    job_id = (request.form.get("job_id") or "").strip()
    url = (request.form.get("url") or "").strip()
    selected_format = (request.form.get("format_selector") or "").strip()
    selected_format_strategy = (request.form.get("format_strategy") or "").strip()
    destination = (request.form.get("destination") or "browser").strip()
    storage_mode = (request.form.get("storage_mode") or "standard").strip()
    drive_folder_id = (request.form.get("drive_folder_id") or "").strip()
    custom_filename = (request.form.get("custom_filename") or "").strip()
    async_drive = (request.form.get("async_drive") or "").strip() == "1"
    cookies_from_browser = (request.form.get("cookies_from_browser") or "").strip()
    allow_fallback = (request.form.get("allow_fallback") or "").strip() == "1"
    prefer_non_hls_auto = (request.form.get("prefer_non_hls_auto") or "0").strip() == "1"
    use_resolution_preset = (request.form.get("use_resolution_preset") or "").strip() == "1"
    batch_auto_title = (request.form.get("batch_auto_title") or "").strip() == "1"
    batch_max_filesize_mb_raw = (request.form.get("batch_max_filesize_mb") or "").strip()
    preferred_resolution = (request.form.get("preferred_resolution") or "auto").strip().lower() or "auto"
    transcribe_subtitles = (request.form.get("transcribe_subtitles") or "").strip() == "1"
    download_chat_danmaku = (request.form.get("download_chat_danmaku") or "").strip() == "1"
    transcription_language = (request.form.get("transcription_language") or "auto").strip().lower() or "auto"
    transcription_model = (request.form.get("transcription_model") or WHISPER_MODEL_NAME).strip() or WHISPER_MODEL_NAME
    transcription_quality = (request.form.get("transcription_quality") or "balanced").strip().lower() or "balanced"
    known_language_codes = {item["code"] for item in WHISPER_SUPPORTED_LANGUAGES}
    known_model_codes = {item["code"] for item in WHISPER_MODEL_OPTIONS}
    known_quality_codes = set(WHISPER_QUALITY_PRESETS.keys())
    if transcription_language not in known_language_codes:
        transcription_language = "auto"
    if transcription_model not in known_model_codes:
        transcription_model = WHISPER_MODEL_NAME
    if transcription_quality not in known_quality_codes:
        transcription_quality = "balanced"
    known_resolution_codes = {item["code"] for item in COMMON_RESOLUTION_OPTIONS}
    if preferred_resolution not in known_resolution_codes:
        preferred_resolution = "auto"
    try:
        batch_max_filesize_mb = float(batch_max_filesize_mb_raw) if batch_max_filesize_mb_raw else 0.0
    except ValueError:
        batch_max_filesize_mb = 0.0
    if batch_max_filesize_mb < 0:
        batch_max_filesize_mb = 0.0
    max_filesize_bytes = int(batch_max_filesize_mb * 1024 * 1024) if batch_max_filesize_mb > 0 else None
    effective_allow_fallback = allow_fallback or use_resolution_preset
    effective_prefer_non_hls_auto = prefer_non_hls_auto or use_resolution_preset
    if selected_format:
        effective_format = selected_format
    elif use_resolution_preset:
        effective_format = _selector_for_preferred_resolution(preferred_resolution, effective_prefer_non_hls_auto)
    else:
        effective_format = SAFE_AUTO_FORMAT_SELECTOR if effective_prefer_non_hls_auto else ""

    if (use_resolution_preset or batch_auto_title) and not selected_format:
        batch_title, batch_selector, batch_strategy = _resolve_batch_metadata(
            url=url,
            preferred_resolution=preferred_resolution,
            max_filesize_bytes=max_filesize_bytes,
            cookies_from_browser=cookies_from_browser,
        )
        if batch_auto_title and not custom_filename and batch_title:
            custom_filename = batch_title
        if use_resolution_preset and batch_selector:
            effective_format = batch_selector
        if not selected_format_strategy and batch_strategy:
            selected_format_strategy = batch_strategy
    whisper_language = None if transcription_language == "auto" else transcription_language

    if not url:
        abort(400, "A YouTube URL is required.")
    _clear_job_cancelled(job_id)
    process_with_subtitles = transcribe_subtitles or download_chat_danmaku

    if destination == "google_drive":
        setup_error = _google_setup_error()
        if setup_error:
            if async_drive:
                return jsonify({"ok": False, "error": setup_error}), 400
            return _render_index(
                url=url,
                error=setup_error,
                selected_destination=destination,
                selected_drive_folder_id=drive_folder_id,
                custom_filename=custom_filename,
                cookies_from_browser=cookies_from_browser,
                allow_fallback="1" if effective_allow_fallback else "0",
                prefer_non_hls_auto="1" if effective_prefer_non_hls_auto else "0",
                transcribe_subtitles="1" if transcribe_subtitles else "0",
                download_chat_danmaku="1" if download_chat_danmaku else "0",
                transcription_language=transcription_language,
                transcription_model=transcription_model,
                transcription_quality=transcription_quality,
                preferred_resolution=preferred_resolution,
            ), 400
        if selected_format and not effective_allow_fallback:
            ok, check_error = _verify_selected_format_available(
                url=url,
                selected_format=selected_format,
                extractor_args_override=selected_format_strategy or None,
                cookies_from_browser=cookies_from_browser,
            )
            if not ok:
                strict_error = (
                    "Selected format is no longer available for this URL/cookies/client context. "
                    "Fetch formats again and retry.\n\n"
                    f"Details: {check_error}"
                )
                if async_drive:
                    return jsonify({"ok": False, "error": strict_error}), 400
                return _render_index(
                    url=url,
                    error=strict_error,
                    selected_destination=destination,
                    selected_drive_folder_id=drive_folder_id,
                    custom_filename=custom_filename,
                    cookies_from_browser=cookies_from_browser,
                    allow_fallback="1" if effective_allow_fallback else "0",
                    prefer_non_hls_auto="1" if effective_prefer_non_hls_auto else "0",
                    transcribe_subtitles="1" if transcribe_subtitles else "0",
                    download_chat_danmaku="1" if download_chat_danmaku else "0",
                    transcription_language=transcription_language,
                    transcription_model=transcription_model,
                    transcription_quality=transcription_quality,
                    preferred_resolution=preferred_resolution,
                ), 400
        try:
            if process_with_subtitles:
                request_id = uuid.uuid4().hex
                source_file = _download_to_local_file(
                    url=url,
                    request_id=request_id,
                    selected_format=effective_format,
                    selected_format_strategy=selected_format_strategy,
                    allow_fallback=effective_allow_fallback,
                    cookies_from_browser=cookies_from_browser,
                    output_dir=TRANSCRIPTION_WORK_DIR,
                    job_id=job_id,
                )
                work_dir = source_file.parent
                subtitle_tracks: list[dict] = []
                cleanup_files: list[Path] = []
                try:
                    transcript_entries: list[dict] = []
                    if transcribe_subtitles:
                        srt_path = work_dir / f"{request_id}.transcript.srt"
                        transcript_lang = _transcribe_to_srt(
                            source_file,
                            srt_path,
                            whisper_language,
                            transcription_model,
                            transcription_quality,
                        )
                        transcript_entries = _parse_srt_entries(srt_path)
                        cleanup_files.append(srt_path)
                        subtitle_tracks.append(
                            {
                                "path": srt_path,
                                "codec": "srt",
                                "language": transcript_lang,
                                "title": "Transcript",
                                "default": not download_chat_danmaku,
                            }
                        )

                    chat_messages: list[dict] = []
                    if download_chat_danmaku:
                        chat_json = _download_live_chat_json3(
                            url=url,
                            request_id=f"{request_id}.livechat",
                            extractor_args_override=selected_format_strategy or None,
                            cookies_from_browser=cookies_from_browser,
                            output_dir=work_dir,
                            job_id=job_id,
                        )
                        chat_messages = _extract_live_chat_messages(chat_json)
                        chat_ass = work_dir / f"{request_id}.chat.ass"
                        _write_danmaku_ass(chat_messages, chat_ass)
                        cleanup_files.extend([chat_json, chat_ass])
                        subtitle_tracks.append(
                            {
                                "path": chat_ass,
                                "codec": "ass",
                                "language": "und",
                                "title": "Live Chat Danmaku",
                                "default": not transcribe_subtitles,
                            }
                        )
                        if transcribe_subtitles and transcript_entries:
                            merged_ass = work_dir / f"{request_id}.merged.ass"
                            _write_danmaku_ass(chat_messages, merged_ass, transcript_entries=transcript_entries)
                            cleanup_files.append(merged_ass)
                            subtitle_tracks.append(
                                {
                                    "path": merged_ass,
                                    "codec": "ass",
                                    "language": transcript_lang if transcribe_subtitles else "und",
                                    "title": "Transcript + Chat",
                                    "default": False,
                                }
                            )

                    muxed_path = work_dir / f"{request_id}.with_subtitles.mkv"
                    _attach_subtitle_tracks(source_file, subtitle_tracks, muxed_path, job_id=job_id)
                    final_filename = _build_final_filename(custom_filename, "mkv")
                    _, web_link = _upload_to_drive(
                        file_path=muxed_path,
                        upload_name=final_filename,
                        folder_id=drive_folder_id or None,
                    )
                finally:
                    for path in [source_file, *cleanup_files, work_dir / f"{request_id}.with_subtitles.mkv"]:
                        try:
                            if path.exists():
                                path.unlink()
                        except Exception:
                            pass
            else:
                ext = _probe_output_extension_with_strategy(
                    url,
                    effective_format,
                    selected_format_strategy or None,
                    cookies_from_browser=cookies_from_browser,
                )
                final_filename = _build_final_filename(custom_filename, ext)
                web_link = None
                last_stream_error = ""
                for attempt_strategy, attempt_format in _download_attempts(
                    effective_format,
                    selected_format_strategy,
                    allow_fallback=effective_allow_fallback,
                ):
                    try:
                        _, web_link = _upload_stream_to_drive(
                            url=url,
                            selected_format=attempt_format,
                            upload_name=final_filename,
                            folder_id=drive_folder_id or None,
                            extractor_args_override=attempt_strategy,
                            cookies_from_browser=cookies_from_browser,
                            job_id=job_id,
                        )
                        break
                    except RuntimeError as stream_err:
                        last_stream_error = str(stream_err)
                        if _is_retryable_download_error(last_stream_error):
                            continue
                        raise
                if web_link is None:
                    raise RuntimeError(last_stream_error or "Download failed.")
        except RuntimeError as exc:
            if async_drive:
                return jsonify({"ok": False, "error": str(exc)}), 400
            return _render_index(
                url=url,
                error=str(exc),
                selected_destination=destination,
                selected_drive_folder_id=drive_folder_id,
                custom_filename=custom_filename,
                cookies_from_browser=cookies_from_browser,
                allow_fallback="1" if effective_allow_fallback else "0",
                prefer_non_hls_auto="1" if effective_prefer_non_hls_auto else "0",
                transcribe_subtitles="1" if transcribe_subtitles else "0",
                download_chat_danmaku="1" if download_chat_danmaku else "0",
                transcription_language=transcription_language,
                transcription_model=transcription_model,
                transcription_quality=transcription_quality,
                preferred_resolution=preferred_resolution,
            ), 400
        except Exception as exc:
            error_text = f"Google Drive upload failed: {exc}"
            if async_drive:
                return jsonify({"ok": False, "error": error_text}), 400
            return _render_index(
                url=url,
                error=error_text,
                selected_destination=destination,
                selected_drive_folder_id=drive_folder_id,
                custom_filename=custom_filename,
                cookies_from_browser=cookies_from_browser,
                allow_fallback="1" if effective_allow_fallback else "0",
                prefer_non_hls_auto="1" if effective_prefer_non_hls_auto else "0",
                transcribe_subtitles="1" if transcribe_subtitles else "0",
                download_chat_danmaku="1" if download_chat_danmaku else "0",
                transcription_language=transcription_language,
                transcription_model=transcription_model,
                transcription_quality=transcription_quality,
                preferred_resolution=preferred_resolution,
            ), 400
        info = f"Uploaded {final_filename} to Google Drive."
        if async_drive:
            return jsonify({"ok": True, "info": info, "drive_link": web_link})
        return _render_index(
            url=url,
            info=info,
            drive_link=web_link,
            selected_destination=destination,
            selected_drive_folder_id=drive_folder_id,
            custom_filename=custom_filename,
            cookies_from_browser=cookies_from_browser,
            allow_fallback="1" if effective_allow_fallback else "0",
            prefer_non_hls_auto="1" if effective_prefer_non_hls_auto else "0",
            transcribe_subtitles="1" if transcribe_subtitles else "0",
            download_chat_danmaku="1" if download_chat_danmaku else "0",
            transcription_language=transcription_language,
            transcription_model=transcription_model,
            transcription_quality=transcription_quality,
            preferred_resolution=preferred_resolution,
        )

    if selected_format and not effective_allow_fallback:
        ok, check_error = _verify_selected_format_available(
            url=url,
            selected_format=selected_format,
            extractor_args_override=selected_format_strategy or None,
            cookies_from_browser=cookies_from_browser,
        )
        if not ok:
            abort(
                400,
                "Selected format is no longer available for this URL/cookies/client context. "
                "Fetch formats again and retry.\n\n"
                f"Details: {check_error}",
            )

    request_id = uuid.uuid4().hex
    try:
        output_file = _download_to_local_file(
            url=url,
            request_id=request_id,
            selected_format=effective_format,
            selected_format_strategy=selected_format_strategy,
            allow_fallback=effective_allow_fallback,
            cookies_from_browser=cookies_from_browser,
            output_dir=TRANSCRIPTION_WORK_DIR if process_with_subtitles else DOWNLOAD_DIR,
            job_id=job_id,
        )
    except JobCancelledError as exc:
        abort(400, str(exc))
    except RuntimeError as exc:
        message = str(exc)
        lowered = message.lower()
        if "timed out" in lowered:
            abort(504, message)
        if "not installed" in lowered:
            abort(500, message)
        abort(400, message)

    cleanup_paths: list[Path] = []
    if process_with_subtitles:
        work_dir = output_file.parent
        subtitle_tracks: list[dict] = []
        cleanup_files: list[Path] = []
        muxed_output = work_dir / f"{request_id}.with_subtitles.mkv"
        try:
            transcript_entries: list[dict] = []
            if transcribe_subtitles:
                subtitle_file = work_dir / f"{request_id}.transcript.srt"
                transcript_lang = _transcribe_to_srt(
                    output_file,
                    subtitle_file,
                    whisper_language,
                    transcription_model,
                    transcription_quality,
                )
                transcript_entries = _parse_srt_entries(subtitle_file)
                cleanup_files.append(subtitle_file)
                subtitle_tracks.append(
                    {
                        "path": subtitle_file,
                        "codec": "srt",
                        "language": transcript_lang,
                        "title": "Transcript",
                        "default": not download_chat_danmaku,
                    }
                )

            chat_messages: list[dict] = []
            if download_chat_danmaku:
                chat_json = _download_live_chat_json3(
                    url=url,
                    request_id=f"{request_id}.livechat",
                    extractor_args_override=selected_format_strategy or None,
                    cookies_from_browser=cookies_from_browser,
                    output_dir=work_dir,
                    job_id=job_id,
                )
                chat_messages = _extract_live_chat_messages(chat_json)
                chat_ass = work_dir / f"{request_id}.chat.ass"
                _write_danmaku_ass(chat_messages, chat_ass)
                cleanup_files.extend([chat_json, chat_ass])
                subtitle_tracks.append(
                    {
                        "path": chat_ass,
                        "codec": "ass",
                        "language": "und",
                        "title": "Live Chat Danmaku",
                        "default": not transcribe_subtitles,
                    }
                )
                if transcribe_subtitles and transcript_entries:
                    merged_ass = work_dir / f"{request_id}.merged.ass"
                    _write_danmaku_ass(chat_messages, merged_ass, transcript_entries=transcript_entries)
                    cleanup_files.append(merged_ass)
                    subtitle_tracks.append(
                        {
                            "path": merged_ass,
                            "codec": "ass",
                            "language": transcript_lang if transcribe_subtitles else "und",
                            "title": "Transcript + Chat",
                            "default": False,
                        }
                    )

            _attach_subtitle_tracks(output_file, subtitle_tracks, muxed_output, job_id=job_id)
        except RuntimeError as exc:
            for path in [*cleanup_files, muxed_output]:
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass
            abort(400, str(exc))
        cleanup_paths.extend([output_file, *cleanup_files])
        output_file = muxed_output
        final_filename = _build_final_filename(custom_filename, "mkv")
    else:
        extension = output_file.suffix.lstrip(".") or "bin"
        final_filename = _build_final_filename(custom_filename, extension)

    if storage_mode != "minimal":
        for path in cleanup_paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

    if transcribe_subtitles:
        return _send_and_cleanup(output_file, final_filename, cleanup_paths)

    return send_file(
        output_file,
        as_attachment=True,
        download_name=final_filename,
        max_age=0,
    ) if storage_mode != "minimal" else _send_and_cleanup(output_file, final_filename, cleanup_paths)


def _send_and_cleanup(output_file: Path, final_filename: str, cleanup_paths: list[Path] | None = None):
    @after_this_request
    def _cleanup(response):
        paths = [output_file]
        if cleanup_paths:
            paths.extend(cleanup_paths)
        try:
            for path in paths:
                if path.exists():
                    path.unlink()
        except Exception:
            pass
        return response

    return send_file(
        output_file,
        as_attachment=True,
        download_name=final_filename,
        max_age=0,
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug_enabled = os.environ.get("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug_enabled)
