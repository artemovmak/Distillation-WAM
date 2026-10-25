# LIBERO-plus results — `public_nas10`

```
policy=lerobot/smolvla_libero
chunk=10
sample_sha256=e687a4a6aa6a86220e4b36fc7a700769262ee8d81be77248914751137ae451b9
seed=1000 n_episodes=1 batch_size=1 env.type=libero
extra=--policy.n_action_steps=10 --rename_map=\{\"observation.images.image\":\ \"observation.images.camera1\"\,\ \"observation.images.image2\":\ \"observation.images.camera2\"\}
```

| metric | value |
|---|---|
| sampled tasks | 280 (K=10, seed=1000) |
| completed / crashed / missing | 280 / 0 / 0 |
| excluded from sampling | 0 (noise kinds []) |
| **SR, completed tasks** | **25.7 [21–31] (n=280)** |
| SR, crashed = failure | 25.7 (72/280) |
| population-weighted SR (est. full benchmark, 10030 eligible tasks, 28 cells) | 22.3 |

## Per suite

| suite | SR % [95% CI] (n) | crashed | weighted % |
|---|---|---|---|
| libero_spatial | 35.7 [26–47] (n=70) | 0 | 34.7 |
| libero_object | 34.3 [24–46] (n=70) | 0 | 29.2 |
| libero_goal | 10.0 [5–19] (n=70) | 0 | 9.2 |
| libero_10 | 22.9 [15–34] (n=70) | 0 | 17.2 |

## Per perturbation axis

| axis | SR % [95% CI] (n) | crashed | weighted % |
|---|---|---|---|
| Camera Viewpoints | 15.0 [7–29] (n=40) | 0 | 13.1 |
| Robot Initial States | 20.0 [10–35] (n=40) | 0 | 21.1 |
| Language Instructions | 15.0 [7–29] (n=40) | 0 | 13.6 |
| Light Conditions | 52.5 [37–67] (n=40) | 0 | 47.4 |
| Background Textures | 35.0 [22–50] (n=40) | 0 | 37.9 |
| Sensor Noise | 12.5 [5–26] (n=40) | 0 | 10.5 |
| Objects Layout | 30.0 [18–45] (n=40) | 0 | 24.8 |

## Per difficulty level (fork difficulty_level)

| level | SR % [95% CI] (n) | crashed |
|---|---|---|
| 1 | 29.8 [20–43] (n=57) | 0 |
| 2 | 28.6 [18–41] (n=56) | 0 |
| 3 | 32.1 [21–45] (n=56) | 0 |
| 4 | 21.4 [13–34] (n=56) | 0 |
| 5 | 16.4 [9–28] (n=55) | 0 |

## Suite × axis (SR %, completed/sampled)

| suite | camera | robot | language | light | background | noise | layout | all |
|---|---|---|---|---|---|---|---|---|
| libero_spatial | 30 (10) | 20 (10) | 10 (10) | 70 (10) | 40 (10) | 40 (10) | 40 (10) | 35.7 |
| libero_object | 10 (10) | 40 (10) | 30 (10) | 70 (10) | 40 (10) | 10 (10) | 40 (10) | 34.3 |
| libero_goal | 0 (10) | 10 (10) | 0 (10) | 20 (10) | 20 (10) | 0 (10) | 20 (10) | 10.0 |
| libero_10 | 20 (10) | 10 (10) | 20 (10) | 50 (10) | 40 (10) | 0 (10) | 20 (10) | 22.9 |
| **all** | 15.0 | 20.0 | 15.0 | 52.5 | 35.0 | 12.5 | 30.0 | 25.7 |

## Axis × level (SR %, n)

| axis | L1 | L2 | L3 | L4 | L5 | NA |
|---|---|---|---|---|---|---|
| camera | 12 (8) | 12 (8) | 25 (8) | 25 (8) | 0 (8) | – |
| robot | 12 (8) | 25 (8) | 62 (8) | 0 (8) | 0 (8) | – |
| language | 25 (8) | 0 (8) | 0 (8) | 25 (8) | 25 (8) | – |
| light | 62 (8) | 38 (8) | 62 (8) | 50 (8) | 50 (8) | – |
| background | 33 (9) | 75 (8) | 38 (8) | 25 (8) | 0 (7) | – |
| noise | 12 (8) | 25 (8) | 0 (8) | 0 (8) | 25 (8) | – |
| layout | 50 (8) | 25 (8) | 38 (8) | 25 (8) | 12 (8) | – |

## Sensor-noise kinds (fork env_wrapper.py:294-307)

| noise kind | SR % [95% CI] (n) | crashed |
|---|---|---|
| motion | 11.1 [2–44] (n=9) | 0 |
| gaussian | 11.1 [2–44] (n=9) | 0 |
| zoom | 0.0 [0–32] (n=8) | 0 |
| fog | 33.3 [10–70] (n=6) | 0 |
| glass | 12.5 [2–47] (n=8) | 0 |

Deviations from the official LIBERO-plus protocol: see RUN.md (subsample, motion blur excluded, instruction tails kept, lerobot 0.4.4 reset path, 256 px rendering).
