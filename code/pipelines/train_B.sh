#!/usr/bin/env bash
# Arm B: SmolVLA + world-model auxiliary loss. Heads on the 64 scene-image tokens predict centred V-JEPA 2 latents of
# the scene frame 5 / 10 / 20 frames ahead (0.5 / 1 / 2 s at 10 Hz), lambda = 0.5. Three steps:
#   1. V-JEPA 2 targets for every scene frame (computed once, frozen encoder)
#   2. training (train_wmaux.py = lerobot-train with the auxiliary heads)
#   3. heads stripped from every checkpoint -> plain SmolVLA for evaluation and deployment (pretrained_model_plain)
#   DATA=<LeRobot dataset root> RUN=<output dir> TARGETS=<dir for V-JEPA 2 latents> [SEED=1000] bash train_B.sh
set -euo pipefail
: "${DATA:?LeRobot dataset root}" "${RUN:?output dir}" "${TARGETS:?targets dir}"
SEED=${SEED:-1000}
HERE=$(cd "$(dirname "$0")/../training" && pwd)

[ -f "$TARGETS/latents.npy" ] || python "$HERE/vjepa_targets.py" --root "$DATA" --out "$TARGETS"

WMAUX_TARGETS="$TARGETS" WMAUX_LAMBDA=0.5 python "$HERE/train_wmaux.py" --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true --policy.prefix_length=0 --policy.pad_language_to=max_length --policy.num_expert_layers=0 \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.push_to_hub=false --dataset.repo_id=local/arm_b --dataset.root="$DATA" \
  --steps=10000 --save_freq=2000 --log_freq=100 --batch_size=64 --num_workers=4 \
  --output_dir="$RUN" --job_name=arm_b --wandb.enable=false --seed="$SEED"

for C in $(ls "$RUN/checkpoints" | grep -E "^[0-9]+$"); do
  python "$HERE/strip_wm_heads.py" "$RUN/checkpoints/$C/pretrained_model" "$RUN/checkpoints/$C/pretrained_model_plain"
done
