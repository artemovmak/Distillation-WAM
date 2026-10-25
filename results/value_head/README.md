# Signal 3: value / outcome head

Features: averaged image tokens of the frozen SmolVLA image encoder (pilot checkpoint, D = 960) plus the 7-d robot
state. Target: episode outcome (success vs failure). Logistic regression and an MLP; 122 pilot episodes, 20,585 frames.

| AUROC | logistic regression | MLP |
|---|---|---|
| validation, after the error moment | 0.997 | 0.998 |
| validation, first frame (should be ≈ 0.5) | 0.964 | 0.845 |
| cross-validation by recording session, after the error moment | 0.43 | 0.40 |
| robot state only, after the error moment | 0.50 | 0.45 |

Near-perfect validation scores come from recognising the recording session: successes and failures were recorded in
blocks by episode type, and train and validation came from the same sessions. On held-out sessions the head is worse
than chance. Consequence for data collection: successes, retries and deliberate failures are interleaved within every
session, and splits are made by session.
