# LIBERO-plus results — `x2_025000_nas10_cleanlang`

```
policy=~/lerobot/runs/x2_libero/checkpoints/025000/pretrained_model
chunk=10
sample_sha256=e687a4a6aa6a86220e4b36fc7a700769262ee8d81be77248914751137ae451b9
seed=1000 n_episodes=1 batch_size=1 env.type=libero
extra=--policy.n_action_steps=10
```

| metric | value |
|---|---|
| sampled tasks | 280 (K=10, seed=1000) |
| completed / crashed / missing | 280 / 0 / 0 |
| excluded from sampling | 0 (noise kinds []) |
| **SR, completed tasks** | **43.9 [38–50] (n=280)** |
| SR, crashed = failure | 43.9 (123/280) |
| population-weighted SR (est. full benchmark, 10030 eligible tasks, 28 cells) | 41.6 |

## Per suite

| suite | SR % [95% CI] (n) | crashed | weighted % |
|---|---|---|---|
| libero_spatial | 51.4 [40–63] (n=70) | 0 | 48.9 |
| libero_object | 57.1 [45–68] (n=70) | 0 | 55.0 |
| libero_goal | 37.1 [27–49] (n=70) | 0 | 35.7 |
| libero_10 | 30.0 [21–42] (n=70) | 0 | 27.2 |

## Per perturbation axis

| axis | SR % [95% CI] (n) | crashed | weighted % |
|---|---|---|---|
| Camera Viewpoints | 47.5 [33–63] (n=40) | 0 | 43.1 |
| Robot Initial States | 45.0 [31–60] (n=40) | 0 | 44.1 |
| Language Instructions | 5.0 [1–17] (n=40) | 0 | 5.3 |
| Light Conditions | 80.0 [65–90] (n=40) | 0 | 77.7 |
| Background Textures | 57.5 [42–71] (n=40) | 0 | 53.3 |
| Sensor Noise | 32.5 [20–48] (n=40) | 0 | 38.3 |
| Objects Layout | 40.0 [26–55] (n=40) | 0 | 42.0 |

## Per difficulty level (fork difficulty_level)

| level | SR % [95% CI] (n) | crashed |
|---|---|---|
| 1 | 56.1 [43–68] (n=57) | 0 |
| 2 | 48.2 [36–61] (n=56) | 0 |
| 3 | 42.9 [31–56] (n=56) | 0 |
| 4 | 35.7 [24–49] (n=56) | 0 |
| 5 | 36.4 [25–50] (n=55) | 0 |

## Suite × axis (SR %, completed/sampled)

| suite | camera | robot | language | light | background | noise | layout | all |
|---|---|---|---|---|---|---|---|---|
| libero_spatial | 70 (10) | 30 (10) | 10 (10) | 90 (10) | 90 (10) | 40 (10) | 30 (10) | 51.4 |
| libero_object | 50 (10) | 70 (10) | 10 (10) | 100 (10) | 50 (10) | 30 (10) | 90 (10) | 57.1 |
| libero_goal | 60 (10) | 60 (10) | 0 (10) | 40 (10) | 40 (10) | 40 (10) | 20 (10) | 37.1 |
| libero_10 | 10 (10) | 20 (10) | 0 (10) | 90 (10) | 50 (10) | 20 (10) | 20 (10) | 30.0 |
| **all** | 47.5 | 45.0 | 5.0 | 80.0 | 57.5 | 32.5 | 40.0 | 43.9 |

## Axis × level (SR %, n)

| axis | L1 | L2 | L3 | L4 | L5 | NA |
|---|---|---|---|---|---|---|
| camera | 75 (8) | 25 (8) | 25 (8) | 62 (8) | 50 (8) | – |
| robot | 62 (8) | 50 (8) | 75 (8) | 25 (8) | 12 (8) | – |
| language | 0 (8) | 12 (8) | 0 (8) | 12 (8) | 0 (8) | – |
| light | 75 (8) | 75 (8) | 100 (8) | 62 (8) | 88 (8) | – |
| background | 67 (9) | 88 (8) | 38 (8) | 25 (8) | 71 (7) | – |
| noise | 62 (8) | 50 (8) | 12 (8) | 25 (8) | 12 (8) | – |
| layout | 50 (8) | 38 (8) | 50 (8) | 38 (8) | 25 (8) | – |

## Sensor-noise kinds (fork env_wrapper.py:294-307)

| noise kind | SR % [95% CI] (n) | crashed |
|---|---|---|
| motion | 33.3 [12–65] (n=9) | 0 |
| gaussian | 22.2 [6–55] (n=9) | 0 |
| zoom | 12.5 [2–47] (n=8) | 0 |
| fog | 66.7 [30–90] (n=6) | 0 |
| glass | 37.5 [14–69] (n=8) | 0 |

Deviations from the official LIBERO-plus protocol: see RUN.md (subsample, motion blur excluded, instruction tails kept, lerobot 0.4.4 reset path, 256 px rendering).
