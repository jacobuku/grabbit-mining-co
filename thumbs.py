"""
thumbs.py — middle-frame thumbnail for every hit in results.json.

Downloads each hit's presigned clip URL (regenerated via /api/v1/videos/playback-url if
it has expired), grabs the frame at half the clip duration with ffmpeg, writes
thumbs/<clip_id>.jpg (640 px wide, JPEG q 5) and sets hit["thumb_url"] to that path.

NOTE: needs a playback URL from the backend for each hit. mine.py no longer writes
hit["video_url"], so this only works for hits that already have a thumbnail on disk;
fresh results need a per-hit playback URL from /api/v1/videos/playback-url.

Usage:
  python thumbs.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from urllib.parse import unquote, urlparse

from mine import TIMEOUT_S, Backend, load_env

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.json")
THUMBS = os.path.join(HERE, "thumbs")
MAX_TOTAL_BYTES = 10 * 1024 * 1024


def source_from_url(url):
    """Presigned URLs are path-style: http://host/<bucket>/<key>?..."""
    return "s3://" + unquote(urlparse(url).path.lstrip("/"))


def download(url, dest):
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def duration_s(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def middle_frame(video, jpg):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{duration_s(video) / 2:.3f}", "-i", video,
         "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "5", jpg],
        check=True)


def main():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe not found on this VM")
    load_env()
    os.makedirs(THUMBS, exist_ok=True)
    with open(RESULTS) as f:
        results = json.load(f)

    backend = None
    hits = [h for run in results["runs"] for h in run["hits"]]
    with tempfile.TemporaryDirectory() as tmp:
        for hit in hits:
            rel = f"thumbs/{hit['id']}.jpg"
            jpg = os.path.join(HERE, rel)
            if not os.path.exists(jpg):
                video = os.path.join(tmp, "clip.mp4")
                try:
                    download(hit["video_url"], video)
                except urllib.error.HTTPError as e:
                    if e.code not in (400, 403):
                        raise
                    if backend is None:
                        backend = Backend()
                        backend.login()
                    hit["video_url"] = backend.playback_url(source_from_url(hit["video_url"]))["url"]
                    print(f"  regenerated URL  {hit['id']}")
                    download(hit["video_url"], video)
                middle_frame(video, jpg)
            hit["thumb_url"] = rel

    total = sum(os.path.getsize(os.path.join(THUMBS, n)) for n in os.listdir(THUMBS))
    print(f"{len(hits)} hits, {len(os.listdir(THUMBS))} thumbs, {total / 1024:.0f} KB total")
    if total > MAX_TOTAL_BYTES:
        sys.exit(f"thumbs/ is {total / 1024 / 1024:.1f} MB, over the 10 MB limit; results.json not updated")

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
