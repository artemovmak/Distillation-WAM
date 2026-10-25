"""SmolVLA + world-model auxiliary loss (arm B-1): lerobot-train with two patched methods.

From the VLM output at the scene image tokens (the first 64 prefix tokens; images come in config.image_features order,
scene first, no special image tokens) small heads predict the V-JEPA 2 latents (vjepa_targets.py) of the scene frame
k frames ahead, k in WMAUX_FUTURES. loss = flow-matching loss + WMAUX_LAMBDA * mean_k smooth_l1(head_k(tokens), LN(target_k - mu)),
mu = the per-token-position mean latent of the training set: raw V-JEPA latents are mostly the static scene (frames of
different episodes have cosine 0.80), centred ones carry the arm and the object (0.01 across episodes, 0.33 -> 0.24 from
0.5 s to 2 s ahead), 2026-10-14.
The heads exist only in training: strip_wm_heads.py removes them, the deployed model is a plain SmolVLA.
All lerobot-train arguments pass through unchanged; the auxiliary settings come from the environment:

    WMAUX_TARGETS=~/data/wm_targets/v4split_train WMAUX_LAMBDA=0.5 python train_wmaux.py --policy.type=smolvla ...
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching, make_att_2d_masks

TARGETS = os.path.expanduser(os.environ["WMAUX_TARGETS"])
LAMBDA = float(os.environ.get("WMAUX_LAMBDA", "0.5"))
FUTURES = [int(k) for k in os.environ.get("WMAUX_FUTURES", "5 10 20").split()]
N_TOK, D_LAT = 64, 1024

LATENTS = torch.from_numpy(np.load(f"{TARGETS}/latents.npy", mmap_mode="r")[:])   # N, 64, 1024 float16
FUTURE = {k: torch.from_numpy(np.load(f"{TARGETS}/future_{k}.npy")).long() for k in FUTURES}
MU = sum(LATENTS[i:i + 2048].double().sum(0) for i in range(0, len(LATENTS), 2048)).div(len(LATENTS)).float()


def target(rows):
    return F.layer_norm(LATENTS[rows].float() - MU, (D_LAT,))


def _baselines():  # loss of trivial predictors on 2000 frames: what the heads must beat
    rows = torch.from_numpy(np.random.default_rng(0).choice(len(LATENTS), min(2000, len(LATENTS)), replace=False)).long()
    now = target(rows)
    out = []
    for k in FUTURES:
        fut = target(FUTURE[k][rows])
        out.append(f"k{k}: zero {F.smooth_l1_loss(torch.zeros_like(fut), fut):.4f} copy-now {F.smooth_l1_loss(now, fut):.4f}")
    return "; ".join(out)


print(f"wm-aux: targets {TARGETS} {tuple(LATENTS.shape)}, lambda {LAMBDA}, futures {FUTURES}, centred targets", flush=True)
print(f"wm-aux baselines: {_baselines()}", flush=True)


def flow_forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    """VLAFlowMatching.forward of lerobot 0.4.4, keeping the VLM (prefix) output for the auxiliary heads."""
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)
    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
        attention_mask=att_2d_masks, position_ids=position_ids, past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, fill_kv_cache=False)
    self._prefix_out = prefix_out
    suffix_out = suffix_out[:, -self.config.chunk_size:].to(dtype=torch.float32)
    v_t = self.action_out_proj(suffix_out)
    return F.mse_loss(u_t, v_t, reduction="none")


_init, _forward = SmolVLAPolicy.__init__, SmolVLAPolicy.forward


def policy_init(self, config, *args, **kwargs):
    _init(self, config, *args, **kwargs)
    d = self.model.vlm_with_expert.vlm.config.text_config.hidden_size
    self.wm_heads = nn.ModuleDict({str(k): nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D_LAT), nn.GELU(), nn.Linear(D_LAT, D_LAT))
                                   for k in FUTURES})


def policy_forward(self, batch, noise=None, time=None, reduction="mean"):
    loss, info = _forward(self, batch, noise, time, reduction)
    if not self.training:
        return loss, info
    prefix = self.model._prefix_out
    assert prefix.shape[1] >= 2 * N_TOK, prefix.shape
    tok = prefix[:, :N_TOK].float()
    idx = batch["index"].long().view(-1).cpu()
    aux = 0.0
    for k in FUTURES:
        tgt = target(FUTURE[k][idx]).to(tok.device, non_blocking=True)
        lk = F.smooth_l1_loss(self.wm_heads[str(k)](tok), tgt, reduction="none").mean(dim=(1, 2))
        info[f"wm_{k}"] = lk.mean().item()
        aux = aux + lk / len(FUTURES)
    info["wm_aux"] = aux.mean().item()
    self._wm_log = getattr(self, "_wm_log", []) + [[info[f"wm_{k}"] for k in FUTURES]]
    if len(self._wm_log) == 100:                     # lerobot's tracker only prints its own keys
        m = np.mean(self._wm_log, axis=0)
        self._wm_steps = getattr(self, "_wm_steps", 0) + 100
        print(f"wm-aux step {self._wm_steps}: " + " ".join(f"k{k} {v:.4f}" for k, v in zip(FUTURES, m)), flush=True)
        self._wm_log = []
    if reduction == "none":
        return loss + LAMBDA * aux, info
    return loss + LAMBDA * aux.mean(), info


VLAFlowMatching.forward = flow_forward
SmolVLAPolicy.__init__ = policy_init
SmolVLAPolicy.forward = policy_forward

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main
    main()
