"""Arm C (pi0.5 -> SmolVLA distillation): lerobot-train for SmolVLA with action targets partly replaced by the teacher.

Per sample: frames of deliberate-failure episodes always get the teacher chunk (their own actions must not be imitated);
other frames get the teacher chunk with probability KD_P_TEACHER and the demonstration chunk otherwise. Teacher chunks
(teacher_label.py, raw units) are normalised with the dataset's action mean/std, exactly as the pre-processor normalises
the demonstration actions (SmolVLA: ACTION -> MEAN_STD). Settings come from the environment:

    KD_TEACHER=~/data/teacher/v4kd_train KD_P_TEACHER=0.5 python train_kd.py --policy.type=smolvla --dataset.root=... ...
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

TEACHER = Path(os.path.expanduser(os.environ["KD_TEACHER"]))
P_TEACHER = float(os.environ.get("KD_P_TEACHER", "0.5"))
info = json.loads((TEACHER / "info.json").read_text())
ROOT = Path(info["root"])
ACT = torch.from_numpy(np.load(TEACHER / "teacher_actions.npy"))                  # N, chunk, 7 raw
stats = json.loads((ROOT / "meta/stats.json").read_text())["action"]
MEAN, STD = torch.tensor(stats["mean"]).float(), torch.tensor(stats["std"]).float()
eps = json.loads((ROOT / "phantom_episodes.json").read_text())
fail_eps = {e["episode_index"] for e in eps if e.get("failure_demo")}
EPISODE = torch.from_numpy(np.load(TEACHER / "episode.npy")).long()                # episode of every global frame
print(f"kd: teacher {TEACHER} {tuple(ACT.shape)}, p_teacher {P_TEACHER}, failure episodes {len(fail_eps)}", flush=True)

_forward = SmolVLAPolicy.forward


def policy_forward(self, batch, noise=None, time=None, reduction="mean"):
    if self.training:
        idx = batch["index"].long().view(-1).cpu()
        epi = EPISODE[idx]
        act = batch["action"]
        chunk = act.shape[1]
        teacher = ((ACT[idx, :chunk] - MEAN) / (STD + 1e-8)).to(act.device, act.dtype)
        is_fail = torch.tensor([int(e) in fail_eps for e in epi.tolist()])
        use = is_fail | (torch.rand(len(idx)) < P_TEACHER)
        u = use.to(act.device)
        batch["action"] = torch.where(u[:, None, None], teacher, act)
        pad = batch.get("actions_id_pad")
        if pad is not None:
            batch["actions_id_pad"] = pad & ~u[:, None]
        self._kd_log = getattr(self, "_kd_log", []) + [float(use.float().mean())]
        if len(self._kd_log) == 100:
            self._kd_steps = getattr(self, "_kd_steps", 0) + 100
            print(f"kd step {self._kd_steps}: teacher share {np.mean(self._kd_log):.3f}", flush=True)
            self._kd_log = []
    return _forward(self, batch, noise, time, reduction)


SmolVLAPolicy.forward = policy_forward

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main
    main()
