#!/usr/bin/env python3
"""Convert PaliGemma base weights to a PI0Pytorch safetensors checkpoint.

Loads Google's pretrained PaliGemma (SigLIP vision + Gemma 3B language model) from
HuggingFace (google/paligemma-3b-pt-224) and produces a PI0Pytorch-compatible
model.safetensors. The action expert (Gemma 300M) and flow-matching projection
heads are left at random init. Mirrors the initialization used to train pi0.5
from scratch.

The HuggingFace model is gated: you must have accepted the license at
https://huggingface.co/google/paligemma-3b-pt-224 and be logged in via
`huggingface-cli login` (or have HF_TOKEN set).

Note: the openpi JAX PaliGemmaWeightLoader points at
gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz, which no longer
supports anonymous access. HF is the practical route.

Usage:
    cd external/openpi
    uv run --active examples/convert_paligemma_to_pytorch.py \
      --output_path ../../egomimic/algo/pi_checkpoints/paligemma_base_pytorch
"""

import json
import os
from typing import Literal

import safetensors.torch
import torch
import tyro
from transformers import PaliGemmaForConditionalGeneration

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch


HF_MODEL_ID = "google/paligemma-3b-pt-224"
KEY_PREFIX = "paligemma_with_expert.paligemma."


def main(
    output_path: str,
    precision: Literal["float32", "bfloat16"] = "bfloat16",
    hf_model_id: str = HF_MODEL_ID,
    action_dim: int = 32,
    action_horizon: int = 100,
    max_token_len: int = 180,
):
    """Download PaliGemma from HF and emit a PI0Pytorch safetensors checkpoint.

    The PaliGemma backbone is initialized from Google's pretrained weights; the
    action expert and projection heads are left at random init.
    """
    print(f"Loading {hf_model_id} from HuggingFace ...")
    hf_model = PaliGemmaForConditionalGeneration.from_pretrained(
        hf_model_id, torch_dtype=torch.float32
    )
    hf_state = hf_model.state_dict()
    print(f"  HF model has {len(hf_state)} tensors")

    prefixed_state = {KEY_PREFIX + k: v for k, v in hf_state.items()}

    model_config = openpi.models.pi0_config.Pi0Config(
        pi05=True,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)

    missing, unexpected = model.load_state_dict(prefixed_state, strict=False)
    print(f"  Loaded  : {len(prefixed_state) - len(unexpected)} PaliGemma tensors")
    print(f"  Random  : {len(missing)} tensors (action expert + projection heads)")
    if unexpected:
        print(f"  WARNING : {len(unexpected)} unexpected keys, first 5:")
        for k in unexpected[:5]:
            print(f"    - {k}")

    # Free the HF model before saving (frees ~5 GB).
    del hf_model, hf_state, prefixed_state

    model = model.to(torch.bfloat16 if precision == "bfloat16" else torch.float32)

    os.makedirs(output_path, exist_ok=True)
    safetensors.torch.save_model(model, os.path.join(output_path, "model.safetensors"))

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(
            {
                "action_dim": action_dim,
                "action_horizon": action_horizon,
                "paligemma_variant": model_config.paligemma_variant,
                "action_expert_variant": model_config.action_expert_variant,
                "precision": precision,
                "init": "paligemma_base",
                "source": f"huggingface:{hf_model_id}",
            },
            f,
            indent=2,
        )

    print(f"Saved to {output_path}")
    print("Action expert and projection heads are randomly initialized.")


if __name__ == "__main__":
    tyro.cli(main)
