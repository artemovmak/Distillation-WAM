#!/usr/bin/env bash
# Arm A: plain SmolVLA, VLM unfrozen. The same recipe as arms B and C, without any extra loss.
#   DATA=<LeRobot dataset root, scene + wrist cameras> RUN=<output dir> [SEED=1000] bash train_A.sh
set -euo pipefail
: "${DATA:?LeRobot dataset root}" "${RUN:?output dir}"
SEED=${SEED:-1000}

lerobot-train --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true --policy.prefix_length=0 --policy.pad_language_to=max_length --policy.num_expert_layers=0 \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.push_to_hub=false --dataset.repo_id=local/arm_a --dataset.root="$DATA" \
  --steps=10000 --save_freq=2000 --log_freq=100 --batch_size=64 --num_workers=4 \
  --output_dir="$RUN" --job_name=arm_a --wandb.enable=false --seed="$SEED"
