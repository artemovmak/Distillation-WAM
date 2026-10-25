"""Wrist camera alignment vs the v2 demo start view (read-only camera use, no arm/gripper motion).
Grabs a C930e frame, compares it with scene_check/wrist_v2_ref.jpg (mean abs diff on 320x180 grey and the brightness of
four horizontal bands, top to bottom: the room shows up as a bright top band), writes wrist_now.jpg and wrist_overlay.jpg.
The arm must be in the task start pose.
    python wrist_align.py
"""
import os

import cv2
import numpy as np

DEV = "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_CD2C619E-video-index0"
out = os.path.expanduser(os.environ.get("SCENE_CHECK_DIR", "~/robot_ws/scene_check")).rstrip("/") + "/"
cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(30):
    ok, now = cap.read()
cap.release()
assert ok, "no frame from the wrist camera (is run_deploy still holding it?)"
ref = cv2.imread(out + "wrist_v2_ref.jpg")
now = cv2.resize(now, (ref.shape[1], ref.shape[0]))
cv2.imwrite(out + "wrist_now.jpg", now)
cv2.imwrite(out + "wrist_overlay.jpg", cv2.addWeighted(ref, 0.5, now, 0.5, 0))
g = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (320, 180)).astype(float)
gr, gn = g(ref), g(now)
rows = lambda im: np.round([im[i * 45:(i + 1) * 45].mean() for i in range(4)]).astype(int).tolist()
print(f"mean abs diff {np.abs(gr - gn).mean():.1f} | brightness ref {gr.mean():.0f} now {gn.mean():.0f} | "
      f"band brightness top->bottom ref {rows(gr)} now {rows(gn)}")
