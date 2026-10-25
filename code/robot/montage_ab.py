"""A vs B comparison video from paired deploy episodes (same cell, same seed): 1920x1080, 2x2 grid
(top: scene camera A | B, bottom: wrist camera A | B), per-pair header, verdict badge at each episode's end,
title and summary cards. Reads the episodes' zarr frames and meta.json only.
    python montage_ab.py OUT.mp4 [--speed 2]
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

R = os.path.expanduser(os.environ.get("ROLLOUTS_DIR", "~/robot_ws/rollouts/20261023")).rstrip("/") + "/"
FPS = 30
W, H = 1920, 1080
PW, PH = 960, 540
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# (cell, A episode, B episode, note) — the last take per cell (operator's choice, 2026-10-23)
PAIRS = [
    (3, "v4_a/ep_student_cube_in_box_v2_1792745751_000", "v4_b/ep_student_cube_in_box_v2_1792745879_000", ""),
    (8, "v4_a/ep_student_cube_in_box_v2_1792745985_000", "v4_b/ep_student_cube_in_box_v2_1792746079_000", ""),
    (2, "v4_a/ep_student_cube_in_box_v2_1792746525_000", "v4_b/ep_student_cube_in_box_v2_1792746597_000", ""),
    (4, "v4_a/ep_student_cube_in_box_v2_1792748055_000", "v4_b/ep_student_cube_in_box_v2_1792748122_000", ""),
    (6, "v4_a/ep_student_cube_in_box_v2_1792747141_000", "v4_b/ep_student_cube_in_box_v2_1792747044_000", ""),
    (9, "v4_a/ep_student_cube_in_box_v2_1792747968_000", "v4_b/ep_student_cube_in_box_v2_1792747888_000", ""),
    (1, "v4_a/ep_student_cube_in_box_v2_1792746164_000", "v4_b/ep_student_cube_in_box_v2_1792746445_000",
     "far edge: both models close the gripper too high"),
]
NAME = {"A": "A  SmolVLA (fine-tuned)", "B": "B  SmolVLA + world-model loss"}
GREEN, RED = (90, 220, 110), (240, 90, 90)


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


class Ep:
    def __init__(self, rel):
        e = R + rel
        self.notes = (json.load(open(e + "/meta.json")).get("notes") or "").replace("operator:", "").strip()
        self.ok = self.notes[:1].lower() == "s"
        self.sd, self.st = self._cam(e, "scene")
        self.wd, self.wt = self._cam(e, "wrist")
        self.t0 = max(self.st[0], self.wt[0])
        self.dur = min(self.st[-1], self.wt[-1]) - self.t0
        self.cache = {}

    @staticmethod
    def _cam(e, cam):
        g = zarr.open(f"{e}/camera_{cam}_color.zarr", mode="r")
        return g["data"], np.asarray(g["ts"][:], float)

    def frame(self, cam, t):
        data, ts = (self.sd, self.st) if cam == "scene" else (self.wd, self.wt)
        i = int(np.clip(np.searchsorted(ts, self.t0 + min(t, self.dur), side="right") - 1, 0, len(ts) - 1))
        key = (cam, i)
        if key not in self.cache:
            raw = data[i]
            im = cv2.imdecode(np.frombuffer(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw), np.uint8),
                              cv2.IMREAD_COLOR)
            self.cache = {k: v for k, v in self.cache.items() if k[0] != cam}
            self.cache[key] = cv2.resize(im, (PW, PH), interpolation=cv2.INTER_AREA)
        return self.cache[key]


def overlay_text(img, items):
    """items: (xy, text, size, fill, bold, anchor) drawn with a dark stroke; returns BGR."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for xy, text, size, fill, bold, anchor in items:
        d.text(xy, text, font=font(size, bold), fill=fill, stroke_width=3, stroke_fill=(0, 0, 0), anchor=anchor)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def badge(img, x, y, ok, secs):
    """Verdict badge (filled rounded box) at panel corner."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    text = f"{'✓ success' if ok else '✗ failure'}  {secs:.0f} s"
    f = font(34, True)
    w = d.textlength(text, font=f)
    d.rounded_rectangle([x, y, x + w + 36, y + 58], radius=12, fill=GREEN if ok else RED)
    d.text((x + 18, y + 9), text, font=f, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def card(lines, secs):
    """Static text card: list of (text, size, bold, colour)."""
    pil = Image.new("RGB", (W, H), (18, 20, 24))
    d = ImageDraw.Draw(pil)
    total = sum(s * 1.55 for _, s, _, _ in lines)
    y = (H - total) / 2
    for text, size, bold, col in lines:
        d.text((W / 2, y), text, font=font(size, bold), fill=col, anchor="ma")
        y += size * 1.55
    im = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return [im] * int(secs * FPS)


def main():
    out = sys.argv[1]
    speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 1.0
    tmp = out + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in card([("World-model supervision for a small VLA policy", 54, True, (240, 240, 240)),
                   ("UR3 · pick up the yellow cube and place it in the left box", 34, False, (190, 190, 190)),
                   ("", 20, False, (0, 0, 0)),
                   ("A  SmolVLA 0.45B, fine-tuned on our demos", 34, False, (200, 200, 200)),
                   ("B  same model and data + auxiliary loss: predict V-JEPA 2 latents of future frames", 34, False, (200, 200, 200)),
                   ("auxiliary heads are removed for deployment: A and B run the identical 0.45B network", 28, False, (150, 150, 150)),
                   ("", 20, False, (0, 0, 0)),
                   ("same cell, same start seed for A and B · " + (f"{speed:g}x speed" if speed != 1 else "real time"),
                    28, False, (150, 150, 150))], 6):
        vw.write(f)
    tally = {"A": 0, "B": 0}
    for cell, ra, rb, note in PAIRS:
        a, b = Ep(ra), Ep(rb)
        tally["A"] += a.ok and cell != 1
        tally["B"] += b.ok and cell != 1
        seg = max(a.dur, b.dur) + 2.0 * speed       # hold the last frames ~2 s of video time
        for k in range(int(seg / speed * FPS)):
            t = k * speed / FPS
            grid = np.vstack([np.hstack([a.frame("scene", t), b.frame("scene", t)]),
                              np.hstack([a.frame("wrist", t), b.frame("wrist", t)])])
            cv2.line(grid, (PW, 0), (PW, H), (255, 255, 255), 3)
            for ep, x0 in ((a, 0), (b, PW)):
                if t >= ep.dur:
                    grid = badge(grid, x0 + 24, PH - 82, ep.ok, ep.dur)
            items = [((24, 20), NAME["A"], 34, (255, 255, 255), True, "la"),
                     ((PW + 24, 20), NAME["B"], 34, (255, 255, 255), True, "la"),
                     ((W / 2, PH + 14), f"cell {cell}", 40, (255, 230, 120), True, "ma"),
                     ((24, H - 24), f"t = {min(t, max(a.dur, b.dur)):4.1f} s", 26, (220, 220, 220), False, "ld"),
                     ((W - 24, H - 24), f"{speed:g}x" if speed != 1 else "1x", 26, (220, 220, 220), False, "rd"),
                     ((24, 64), "scene camera", 22, (200, 200, 200), False, "la"),
                     ((24, PH + 14), "wrist camera", 22, (200, 200, 200), False, "la")]
            if note:
                items.append(((W / 2, H - 24), note, 30, (255, 230, 120), False, "md"))
            vw.write(overlay_text(grid, items))
        print(f"cell {cell}: A {'s' if a.ok else 'f'} {a.dur:.1f}s | B {'s' if b.ok else 'f'} {b.dur:.1f}s", flush=True)
    for f in card([("Results on the real robot", 50, True, (240, 240, 240)),
                   ("", 16, False, (0, 0, 0)),
                   (f"cells 2, 3, 4, 6, 8, 9:   A {tally['A']}/6   ·   B {tally['B']}/6", 40, True, (255, 230, 120)),
                   ("A succeeded on cells 3 and 8 only after dropping and re-grasping the cube", 30, False, (200, 200, 200)),
                   ("cell 1 (far edge): both fail, the gripper closes too high", 30, False, (200, 200, 200)),
                   ("", 16, False, (0, 0, 0)),
                   ("earlier session (Oct 21, mixed cells):   A 1/6   ·   B 7/11", 34, False, (200, 200, 200)),
                   ("offline grounding on held-out demos:   A 0.77   ·   B 0.90   ·   pi0.5 (3.6B) 0.91", 34, False, (200, 200, 200))],
                  7):
        vw.write(f)
    vw.release()
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "22", "-preset", "medium", "-movflags", "+faststart", out],
                       capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode == 0:
        os.remove(tmp)
    else:
        print("ffmpeg failed, kept", tmp, r.stderr.decode()[-300:])
    print("wrote", out, f"{os.path.getsize(out if r.returncode == 0 else tmp) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
