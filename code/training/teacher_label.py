"""Arm C, step 1: the pi0.5 teacher labels every frame of a LeRobot dataset with an action chunk.

Output <out>/teacher_actions.npy float32 [N, chunk, 7] in RAW action units (after the teacher's post-processor), row =
the dataset's global frame index, plus <out>/info.json. Noise seed = batch number, so a re-run reproduces the labels.

    python teacher_label.py --ckpt ~/lerobot/runs/v4s_pi05/checkpoints/008000/pretrained_model \
        --root ~/lerobot/data/makar_v4kd_wrist/train --out ~/data/teacher/v4kd_train
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--chunk", type=int, default=50)
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

cfg = PreTrainedConfig.from_pretrained(a.ckpt)
policy = get_policy_class(cfg.type).from_pretrained(a.ckpt).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt,
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}},
                                     postprocessor_overrides={"device_processor": {"device": "cuda"}})
ds = LeRobotDataset("local/" + a.root.parent.name, root=a.root)
n = len(ds)
assert (np.asarray(ds.hf_dataset["index"]) == np.arange(n)).all()
keep = ("observation.images.scene", "observation.images.wrist", "observation.state", "task")
loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=6, shuffle=False)
out = np.lib.format.open_memmap(a.out / "teacher_actions.npy", mode="w+", dtype=np.float32, shape=(n, a.chunk, 7))
row = 0
with torch.no_grad():
    for b, batch in enumerate(loader):
        x = pre({k: batch[k] for k in keep})
        torch.manual_seed(b)
        act = policy.predict_action_chunk(x)[:, :a.chunk]                     # B, chunk, 7 (normalised)
        act = torch.stack([post(act[:, i, :]) for i in range(act.shape[1])], 1)
        out[row:row + len(act)] = act.float().cpu().numpy()
        row += len(act)
        if b % 100 == 0:
            print(f"{row}/{n}", flush=True)
out.flush()
assert row == n
np.save(a.out / "episode.npy", np.asarray(ds.hf_dataset["episode_index"], dtype=np.int32))
(a.out / "info.json").write_text(json.dumps({"ckpt": a.ckpt, "root": str(a.root), "frames": n, "chunk": a.chunk}, indent=1))
print(f"done: {n} frames -> {a.out}")
