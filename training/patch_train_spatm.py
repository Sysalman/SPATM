#!/usr/bin/env python
"""
patch_train_spatm.py
Hardens train_spatm.py's custom CLIP text-encoder forward against
transformers 5.x internal changes:
  1. position_ids buffer may not exist  -> compute via torch.arange fallback
  2. modeling_attn_mask_utils helpers moved -> guarded import with local fallback
Run once on the instance, in the folder containing train_spatm.py:
    python patch_train_spatm.py
A backup is written to train_spatm.py.bak
"""
import shutil, sys, os

P = "train_spatm.py"
if not os.path.exists(P):
    sys.exit(f"ERROR: {P} not found in this folder. cd to where it lives.")

shutil.copyfile(P, P + ".bak")
lines = open(P, "r").read().split("\n")
out = []
n_pos = n_imp = 0

for ln in lines:
    s = ln.strip()
    indent = ln[:len(ln) - len(ln.lstrip())]

    if s == "position_ids = text_encoder.text_model.embeddings.position_ids[:, :seq_length]":
        n_pos += 1
        out += [
            indent + "_emb = text_encoder.text_model.embeddings",
            indent + "if hasattr(_emb, 'position_ids') and _emb.position_ids is not None:",
            indent + "    position_ids = _emb.position_ids[:, :seq_length]",
            indent + "else:",
            indent + "    position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0)",
        ]
        continue

    if s == "from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask":
        n_imp += 1
        out += [
            indent + "try:",
            indent + "    from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask",
            indent + "except ImportError:",
            indent + "    def _create_4d_causal_attention_mask(input_shape, dtype, device, *a, **k):",
            indent + "        b, t = input_shape",
            indent + "        m = torch.full((t, t), torch.finfo(dtype).min, device=device, dtype=dtype)",
            indent + "        m = torch.triu(m, diagonal=1)",
            indent + "        return m[None, None].expand(b, 1, t, t)",
            indent + "    def _prepare_4d_attention_mask(mask, dtype, *a, **k):",
            indent + "        m = mask[:, None, None, :].to(dtype)",
            indent + "        return (1.0 - m) * torch.finfo(dtype).min",
        ]
        continue

    out.append(ln)

open(P, "w").write("\n".join(out))
print(f"Patched {P}: position_ids fixes={n_pos}, attn-mask-import fixes={n_imp}")
print(f"Backup: {P}.bak")
if n_pos == 0 or n_imp == 0:
    print("WARNING: expected lines not found — file may already be patched or differ.")