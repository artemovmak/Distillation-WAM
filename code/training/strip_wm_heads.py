"""Copy a train_wmaux.py checkpoint (pretrained_model dir) without the wm_heads.* tensors: a plain SmolVLA for deployment.

    python strip_wm_heads.py <run>/checkpoints/010000/pretrained_model <out dir>
"""
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
shutil.copytree(src, dst, dirs_exist_ok=True)
sd = load_file(src / "model.safetensors")
kept = {k: v for k, v in sd.items() if not k.startswith("wm_heads.")}
save_file(kept, dst / "model.safetensors", metadata={"format": "pt"})
print(f"{len(sd) - len(kept)} wm_heads tensors removed, {len(kept)} kept -> {dst}")
