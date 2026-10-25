"""Render deploy episodes to mp4: scene camera | wrist camera side by side, real-time, with a caption
(model label, object, operator verdict/notes, elapsed time). Reads only the episode's zarr frames and meta.json.
    python render_episode.py OUT_DIR EPISODE_DIR [EPISODE_DIR ...]
    python render_episode.py OUT_DIR rollouts/20261023/v4_b/ep_*        (skips episodes already rendered)
    --notes   also print the operator's note (Russian) under the verdict; default shows only success / failure
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

H = 540  # output height per panel; 16:9 panels -> 1920x540 video
FPS = 30
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def frames(ep, cam):
    g = zarr.open(os.path.join(ep, f"camera_{cam}_color.zarr"), mode="r")
    return g["data"], np.asarray(g["ts"][:], dtype=float)


def decode(raw):
    buf = np.frombuffer(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw), np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.resize(im, (H * 16 // 9, H))


def caption(meta):
    tags = meta.get("tags") or {}
    if isinstance(tags, list):
        tags = dict(t.split(":", 1) for t in tags if ":" in t)
    label = tags.get("label") or meta.get("label") or "?"
    model = {"A": "A: SmolVLA", "B": "B: SmolVLA + world-model aux", "C": "C: SmolVLA distilled from pi0.5",
             "P": "pi0.5"}.get(label[:1], label)
    notes = str(meta.get("notes") or "").replace("operator:", "").strip()
    return f"{model}   [{label}, {tags.get('obj', 'cube')}]", notes


def put(img, text, org, scale, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if scale < 0.8 else 2, cv2.LINE_AA)


def overlay(w, title, notes, show_notes):
    """Static captions drawn once with PIL (DejaVu has Cyrillic); returns (BGR image, mask)."""
    im = Image.new("RGB", (w, H)); m = Image.new("L", (w, H))
    d, dm = ImageDraw.Draw(im), ImageDraw.Draw(m)
    def txt(xy, t, size, fill, bold=False):
        f = ImageFont.truetype(FONT_B if bold else FONT, size)
        d.text(xy, t, font=f, fill=fill, stroke_width=3, stroke_fill=(0, 0, 0))
        dm.text(xy, t, font=f, fill=255, stroke_width=3, stroke_fill=255)
    pw = H * 16 // 9
    txt((16, 12), title, 30, (255, 255, 255), bold=True)
    txt((16, 52), "scene camera", 18, (200, 200, 200))
    txt((pw + 16, 52), "wrist camera", 18, (200, 200, 200))
    v = notes[:1].lower()
    if v in ("s", "f"):
        ok = v == "s"
        txt((pw + 16, H - 48 - (30 if show_notes else 0)), "success" if ok else "failure", 30,
            (120, 255, 120) if ok else (255, 120, 120), bold=True)
    if show_notes and notes:
        txt((pw + 16, H - 40), notes[:80], 20, (230, 230, 230))
    bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return bgr, np.asarray(m) > 0


def render(ep, out_dir, show_notes=False):
    name = "_".join(os.path.normpath(ep).split(os.sep)[-2:]) + ".mp4"
    out = os.path.join(out_dir, name)
    if os.path.exists(out):
        return out, "exists"
    meta = json.load(open(os.path.join(ep, "meta.json")))
    title, notes = caption(meta)
    sd, st = frames(ep, "scene")
    wd, wt = frames(ep, "wrist")
    t0, t1 = max(st[0], wt[0]), min(st[-1], wt[-1])
    tmp = out + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (2 * H * 16 // 9, H))
    ov, mask = overlay(2 * H * 16 // 9, title, notes, show_notes)
    si = wi = -1
    s_img = w_img = None
    for t in np.arange(t0, t1, 1.0 / FPS):
        j = min(np.searchsorted(st, t, side="right") - 1, len(st) - 1)
        k = min(np.searchsorted(wt, t, side="right") - 1, len(wt) - 1)
        if j != si:
            s_img, si = decode(sd[max(j, 0)]), j
        if k != wi:
            w_img, wi = decode(wd[max(k, 0)]), k
        f = np.hstack([s_img, w_img])
        f[mask] = ov[mask]
        put(f, f"t = {t - t0:5.1f} s", (16, H - 20), 0.7)
        vw.write(f)
    vw.release()
    # H.264 so the file plays everywhere (QuickTime, Telegram, slides); keep the mp4v file if ffmpeg is missing
    # -nostdin: under `timeout` (background process group) ffmpeg reading the tty gets SIGTTIN and stops forever
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "23", "-preset", "veryfast", out], capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode == 0:
        os.remove(tmp)
    else:
        os.replace(tmp, out)
    return out, f"{t1 - t0:.1f} s, {len(st)} scene / {len(wt)} wrist frames"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--notes"]
    out_dir = args[0]
    os.makedirs(out_dir, exist_ok=True)
    for ep in args[1:]:
        if os.path.isdir(ep):
            print(*render(ep, out_dir, show_notes="--notes" in sys.argv))
