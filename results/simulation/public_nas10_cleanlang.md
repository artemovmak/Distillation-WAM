# LIBERO-plus results — `public_nas10_cleanlang`

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
| **SR, completed tasks** | **48.9 [43–55] (n=280)** |
| SR, crashed = failure | 48.9 (137/280) |
| population-weighted SR (est. full benchmark, 10030 eligible tasks, 28 cells) | 44.9 |

## Per suite

| suite | SR % [95% CI] (n) | crashed | weighted % |
|---|---|---|---|
| libero_spatial | 57.1 [45–68] (n=70) | 0 | 53.1 |
| libero_object | 50.0 [39–61] (n=70) | 0 | 43.7 |
| libero_goal | 47.1 [36–59] (n=70) | 0 | 43.5 |
| libero_10 | 41.4 [31–53] (n=70) | 0 | 39.7 |

## Per perturbation axis

| axis | SR % [95% CI] (n) | crashed | weighted % |
|---|---|---|---|
| Camera Viewpoints | 42.5 [29–58] (n=40) | 0 | 39.1 |
| Robot Initial States | 55.0 [40–69] (n=40) | 0 | 53.7 |
| Language Instructions | 12.5 [5–26] (n=40) | 0 | 10.2 |
| Light Conditions | 75.0 [60–86] (n=40) | 0 | 72.6 |
| Background Textures | 70.0 [55–82] (n=40) | 0 | 74.8 |
| Sensor Noise | 45.0 [31–60] (n=40) | 0 | 43.0 |
| Objects Layout | 42.5 [29–58] (n=40) | 0 | 37.1 |

## Per difficulty level (fork difficulty_level)

| level | SR % [95% CI] (n) | crashed |
|---|---|---|
| 1 | 70.2 [57–80] (n=57) | 0 |
| 2 | 51.8 [39–64] (n=56) | 0 |
| 3 | 50.0 [37–63] (n=56) | 0 |
| 4 | 42.9 [31–56] (n=56) | 0 |
| 5 | 29.1 [19–42] (n=55) | 0 |

## Suite × axis (SR %, completed/sampled)

| suite | camera | robot | language | light | background | noise | layout | all |
|---|---|---|---|---|---|---|---|---|
| libero_spatial | 60 (10) | 70 (10) | 10 (10) | 70 (10) | 80 (10) | 50 (10) | 60 (10) | 57.1 |
| libero_object | 20 (10) | 70 (10) | 30 (10) | 80 (10) | 60 (10) | 40 (10) | 50 (10) | 50.0 |
| libero_goal | 50 (10) | 40 (10) | 0 (10) | 80 (10) | 70 (10) | 60 (10) | 30 (10) | 47.1 |
| libero_10 | 40 (10) | 40 (10) | 10 (10) | 70 (10) | 70 (10) | 30 (10) | 30 (10) | 41.4 |
| **all** | 42.5 | 55.0 | 12.5 | 75.0 | 70.0 | 45.0 | 42.5 | 48.9 |

## Axis × level (SR %, n)

| axis | L1 | L2 | L3 | L4 | L5 | NA |
|---|---|---|---|---|---|---|
| camera | 88 (8) | 38 (8) | 12 (8) | 38 (8) | 38 (8) | – |
| robot | 88 (8) | 62 (8) | 88 (8) | 25 (8) | 12 (8) | – |
| language | 25 (8) | 0 (8) | 0 (8) | 12 (8) | 25 (8) | – |
| light | 88 (8) | 62 (8) | 75 (8) | 88 (8) | 62 (8) | – |
| background | 78 (9) | 100 (8) | 88 (8) | 50 (8) | 29 (7) | – |
| noise | 50 (8) | 50 (8) | 38 (8) | 62 (8) | 25 (8) | – |
| layout | 75 (8) | 50 (8) | 50 (8) | 25 (8) | 12 (8) | – |

## Sensor-noise kinds (fork env_wrapper.py:294-307)

| noise kind | SR % [95% CI] (n) | crashed |
|---|---|---|
| motion | 44.4 [19–73] (n=9) | 0 |
| gaussian | 44.4 [19–73] (n=9) | 0 |
| zoom | 25.0 [7–59] (n=8) | 0 |
| fog | 83.3 [44–97] (n=6) | 0 |
| glass | 37.5 [14–69] (n=8) | 0 |

Deviations from the official LIBERO-plus protocol: see RUN.md (subsample, motion blur excluded, instruction tails kept, lerobot 0.4.4 reset path, 256 px rendering).
