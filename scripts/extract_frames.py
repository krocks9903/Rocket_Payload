"""
Pull numbered frames out of the flight video so terrain_model.py can read them.

Frames are saved as data/frames/f001.jpg, f002.jpg, ... (the names the model and results use).

python scripts/extract_frames.py data/video/flight.mp4 [--start 0.7] [--end 12] [--every 1] [--out data/frames]

  --start / --end : seconds into the video to keep (skip the pad / post-apogee footage)
  --every         : keep every Nth video frame (1 = all of them)

Getting the video: pip install yt-dlp, then
  yt-dlp -f "bv*[ext=mp4]" -o data/video/flight.mp4 <youtube url>
"""

import argparse
import os

import cv2

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--start", type=float, default=0.0)
p.add_argument("--end", type=float, default=None)
p.add_argument("--every", type=int, default=1)
p.add_argument("--out", default="data/frames")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"can't open {args.video}")
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

i, saved = 0, 0
while True:
    ok, frame = cap.read()
    if not ok or (args.end is not None and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > args.end):
        break
    if i % args.every == 0:
        saved += 1
        cv2.imwrite(os.path.join(args.out, f"f{saved:03d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    i += 1
cap.release()
print(f"{saved} frames -> {args.out}  (video {fps:.1f} fps)")
