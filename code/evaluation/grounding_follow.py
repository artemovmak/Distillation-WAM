"""Does the policy decide WHERE to go from the image? Test at the START of the episode (arm at the start pose, so
the arm state says nothing about the cube): plan the first H steps from frame START with the episode's own images vs
the images of the partner episode whose cube (grasp location) is farthest away, keeping the own arm state.
follow = <d_plan, d_gt> / |d_gt|^2, where d_gt = partner's real first-H-step displacement - own one:
1 = the plan moves exactly like the demos do when the cube moves, 0 = the image is ignored.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

def close_index(grip, thr=0.45):
    """Grasp frame: the first frame where the gripper command crosses `thr` upwards (from open). If it never does
    (objects whose grasp command stays lower), the first upward crossing of the mid-level between the episode's
    min and max command. None when the gripper never closes. On the cube data the fixed rule finds every grasp."""
    g = np.asarray(grip, dtype=np.float64)
    for level in (thr, (g.min() + g.max()) / 2 if g.max() - g.min() > 0.1 else None):
        if level is None:
            break
        up = np.nonzero((g[:-1] < level) & (g[1:] >= level))[0]
        if len(up):
            return int(up[0] + 1)
    return None


def load_close_index():
    return close_index

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--start", type=int, default=5)
ap.add_argument("--horizon", type=int, default=30)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

root = a.data.expanduser()
ds = LeRobotDataset(repo_id="local/ds", root=root)
eps = json.loads((root / "phantom_episodes.json").read_text())
froms, tos = ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"]
close_index, H = load_close_index(), a.horizon
frames = []
for rec in eps:
    ei = rec["episode_index"]
    lo, hi = int(froms[ei]), int(tos[ei])
    grip = np.array([float(ds.hf_dataset[i]["action"][6]) for i in range(lo, hi)])
    ci = close_index(grip)
    if ci is None or a.start + H > hi - lo:
        continue
    k = lo + a.start
    acts = np.stack([np.asarray(ds.hf_dataset[i]["action"]) for i in range(k, k + H)]).astype(np.float64)
    frames.append({"ep": ei, "frame": k, "task": rec["task_text"], "gt": acts[:, :3].sum(0) * 1000,
                   "grasp": np.asarray(ds.hf_dataset[lo + ci]["observation.state"], dtype=np.float64)[:3]})
for f in frames:
    o = max((o for o in frames if o is not f), key=lambda o: np.linalg.norm(o["grasp"] - f["grasp"]))
    f["partner"] = o

pol = get_policy_class(PreTrainedConfig.from_pretrained(a.ckpt).type).from_pretrained(a.ckpt).to("cuda").eval()
pre, post = make_pre_post_processors(pol.config, pretrained_path=a.ckpt)
keys = [k for k in pol.config.input_features if k.startswith("observation.images.")]
cache = {}


def sample(i):
    if i not in cache:
        cache[i] = ds[i]
    return cache[i]


def plan(img_src, state_src, task, black=()):
    out = []
    for s in a.seeds:
        b = {k: (torch.zeros_like(sample(img_src[k])[k]) if k in black else sample(img_src[k])[k]) for k in keys}
        b["observation.state"] = sample(state_src)["observation.state"]
        b["task"] = task
        b = pre(b)
        torch.manual_seed(s)
        with torch.no_grad():
            act = post(pol.predict_action_chunk(b))[0, :H].float().cpu().numpy()
        out.append(act[:, :3].sum(0) * 1000)
    return np.mean(out, axis=0), float(np.std(out, axis=0).mean())


rows = []
scene, wrist = "observation.images.scene", "observation.images.wrist"
for f in frames:
    p = f["partner"]
    own = {k: f["frame"] for k in keys}
    real, spread = plan(own, f["frame"], f["task"])
    conds = {"scene_swap": {**own, scene: p["frame"]}, "both_swap": {k: p["frame"] for k in keys}}
    if wrist in keys:
        conds["wrist_swap"] = {**own, wrist: p["frame"]}
    d_gt = p["gt"] - f["gt"]
    r = {"ep": f["ep"], "partner": p["ep"], "cube_shift_mm": float(np.linalg.norm(p["grasp"] - f["grasp"]) * 1000),
         "d_gt_mm": float(np.linalg.norm(d_gt)), "seed_spread_mm": spread,
         "err_real_vs_gt_mm": float(np.linalg.norm(real - f["gt"]))}
    for c, src in conds.items():
        sw, _ = plan(src, f["frame"], f["task"])
        dp = sw - real
        r[c + "_follow"] = float(np.dot(dp, d_gt) / max(np.dot(d_gt, d_gt), 1e-9))
        r[c + "_dplan_mm"] = float(np.linalg.norm(dp))
    rows.append(r)
mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("ep", "partner")}
print(json.dumps({"n": len(rows), "start": a.start, "horizon": H, **{k: round(v, 3) for k, v in mean.items()}}))
a.out.write_text(json.dumps({"mean": mean, "rows": rows}, indent=1))
