"""Pose-independent wrist camera mount check: where the two orange finger pads sit in the wrist frame.
The camera is fixed to the gripper, so the midpoint between the pads (independent of the aperture) and the pads'
top edge give the camera's yaw / pitch relative to the gripper. Compares the live frame with a reference frame.
    python finger_pads.py [reference jpg in scene_check/]   (default wrist_ref_1014_start.jpg)
"""
import sys

import os

import cv2
import numpy as np

out = os.path.expanduser(os.environ.get("SCENE_CHECK_DIR", "~/robot_ws/scene_check")).rstrip("/") + "/"
DEV = "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_CD2C619E-video-index0"


def pads(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (5, 120, 120), (25, 255, 255))
    m[: img.shape[0] // 2] = 0                      # the pads are in the lower half; the room above has orange things too
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    idx = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:2]
    if len(idx) < 2:
        return None
    a, b = sorted(idx, key=lambda i: cen[i][0])
    top = min(st[a, cv2.CC_STAT_TOP], st[b, cv2.CC_STAT_TOP])
    return {"left_x": int(cen[a][0]), "right_x": int(cen[b][0]), "mid_x": int((cen[a][0] + cen[b][0]) / 2), "top_y": int(top),
            "area": int(st[a, cv2.CC_STAT_AREA] + st[b, cv2.CC_STAT_AREA])}


ref = cv2.imread(out + (sys.argv[1] if len(sys.argv) > 1 else "wrist_ref_1014_start.jpg"))
cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(30):
    ok, now = cap.read()
cap.release()
assert ok, "no frame from the wrist camera (is run_deploy still holding it?)"
now = cv2.resize(now, (ref.shape[1], ref.shape[0]))
cv2.imwrite(out + "wrist_now.jpg", now)
cv2.imwrite(out + "wrist_overlay.jpg", cv2.addWeighted(ref, 0.5, now, 0.5, 0))
pr, pn = pads(ref), pads(now)
print("reference pads:", pr)
print("live pads:     ", pn)
if pr and pn:
    dx, dy = pn["mid_x"] - pr["mid_x"], pn["top_y"] - pr["top_y"]
    print(f"pad midpoint shift {dx:+d} px horizontally, pad top {dy:+d} px vertically (1280x720 frame)")
    print(("-> turn the camera " + ("RIGHT" if dx > 0 else "LEFT") + f" ({abs(dx)} px)") if abs(dx) > 25 else "-> horizontal OK")
    print(("-> tilt the camera " + ("UP" if dy > 0 else "DOWN") + f" ({abs(dy)} px)") if abs(dy) > 25 else "-> vertical OK")
