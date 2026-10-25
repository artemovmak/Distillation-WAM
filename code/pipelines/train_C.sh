#!/usr/bin/env bash
# Arm C (control): SmolVLA distilled from pi0.5. The teacher labels every frame with an action chunk; the student sees
# the teacher chunk on every deliberate-failure frame and the teacher or the demonstration with probability 0.5 elsewhere.
# The student dataset includes the failure demos; action/state normalisation = arm A's statistics.
#   DATA=<LeRobot dataset root incl. failure demos> TEACHER_CKPT=<pi0.5 pretrained_model dir> TEACH=<dir for teacher labels>
#   RUN=<output dir> [SEED=1000] bash train_C.sh
set -euo pipefail
: "${DATA:?LeRobot dataset root}" "${RUN:?output dir}" "${TEACHER_CKPT:?pi0.5 checkpoint}" "${TEACH:?teacher labels dir}"
SEED=${SEED:-1000}
HERE=$(cd "$(dirname "$0")/../training" && pwd)

# normalise with arm A's statistics (A_STATS = arm A dataset's meta/stats.json); the dataset's own stats are kept aside
if [ -n "${A_STATS:-}" ] && [ ! -f "$DATA/meta/stats_own.json" ]; then
  cp "$DATA/meta/stats.json" "$DATA/meta/stats_own.json" && cp "$A_STATS" "$DATA/meta/stats.json"
fi

# pi0.5 needs its own environment (openpi-compatible transformers); PI_PYTHON points at it
[ -f "$TEACH/teacher_actions.npy" ] || "${PI_PYTHON:-python}" "$HERE/teacher_label.py" --ckpt "$TEACHER_CKPT" --root "$DATA" --out "$TEACH"

KD_TEACHER="$TEACH" KD_P_TEACHER=0.5 python "$HERE/train_kd.py" --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true --policy.prefix_length=0 --policy.pad_language_to=max_length --policy.num_expert_layers=0 \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.push_to_hub=false --dataset.repo_id=local/arm_c --dataset.root="$DATA" \
  --steps=10000 --save_freq=2000 --log_freq=100 --batch_size=64 --num_workers=4 \
  --output_dir="$RUN" --job_name=arm_c --wandb.enable=false --seed="$SEED"
