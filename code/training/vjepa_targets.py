"""V-JEPA 2 latents of every scene frame of a LeRobot dataset: the targets of the world-model auxiliary loss (arm B-1).

Frame t is encoded from the 2-frame clip (t-1, t) (V-JEPA 2 tubelets span 2 frames; the first frame of an episode is
repeated), 224 -> 256 px, ViT-L/16 -> 16x16 tokens x 1024, average-pooled 2x2 to the 8x8 grid of SmolVLA's image
tokens. Output in <out>/: latents.npy float16 [N, 64, 1024] (row = the dataset's global frame index),
episode.npy int32 [N], and future_<k>.npy int32 [N] = the row k frames later in the same episode (clamped to its end).

    python vjepa_targets.py --root ~/lerobot/data/makar_v4split_wrist/train --out ~/data/wm_targets/v4split_train
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import VJEPA2Model

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--camera", default="observation.images.scene")
ap.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
ap.add_argument("--futures", type=int, nargs="+", default=[5, 10, 20])
ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

ds = LeRobotDataset(repo_id="local/" + a.root.parent.name, root=a.root)
n = len(ds)
episode = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int32)
index = np.asarray(ds.hf_dataset["index"], dtype=np.int64)
assert (index == np.arange(n)).all(), "expected contiguous global indices"

dev = "cuda"
model = VJEPA2Model.from_pretrained(a.model, torch_dtype=torch.float16).to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)
loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=6, shuffle=False)
out = np.lib.format.open_memmap(a.out / "latents.npy", mode="w+", dtype=np.float16, shape=(n, 64, 1024))

prev_frame, prev_ep, row = None, None, 0
with torch.no_grad():
    for batch in loader:
        img = batch[a.camera].to(dev)                                   # B, 3, 224, 224 in [0, 1]
        eps = batch["episode_index"].tolist()
        prev = torch.empty_like(img)
        for i in range(len(eps)):                                       # frame t-1 of the same episode (or t itself)
            if i > 0 and eps[i - 1] == eps[i]:
                prev[i] = img[i - 1]
            elif i == 0 and prev_ep == eps[0]:
                prev[i] = prev_frame
            else:
                prev[i] = img[i]
        prev_frame, prev_ep = img[-1].clone(), eps[-1]
        clip = torch.stack([prev, img], 1)                              # B, 2, 3, 224, 224
        clip = F.interpolate(clip.flatten(0, 1), size=256, mode="bilinear", align_corners=False).view(-1, 2, 3, 256, 256)
        clip = ((clip - mean) / std).half()
        tok = model(pixel_values_videos=clip, skip_predictor=True).last_hidden_state   # B, 256, 1024
        tok = tok.view(-1, 16, 16, 1024).permute(0, 3, 1, 2)
        tok = F.avg_pool2d(tok.float(), 2).permute(0, 2, 3, 1).reshape(-1, 64, 1024)
        out[row:row + len(eps)] = tok.half().cpu().numpy()
        row += len(eps)
        if row % (a.batch * 40) < a.batch:
            print(f"{row}/{n}", flush=True)
out.flush()
assert row == n
np.save(a.out / "episode.npy", episode)
last = {}
for i in range(n):
    last[episode[i]] = i
for k in a.futures:
    fut = np.minimum(np.arange(n) + k, np.array([last[e] for e in episode]))
    np.save(a.out / f"future_{k}.npy", fut.astype(np.int32))
print(f"done: {n} frames, {len(last)} episodes -> {a.out}")
