#!/usr/bin/env python3
"""Convert raw PaliGemma base weights (pt_224.npz) to a PI0Pytorch safetensors checkpoint.

Loads Google's pretrained PaliGemma (SigLIP vision encoder + Gemma 3B language model)
from gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz and produces a
PI0Pytorch-compatible model.safetensors. The action expert (Gemma 300M) and the
flow-matching projection heads are left at random init. This mirrors the
initialization used to train pi0.5 from scratch.

Usage:
    cd external/openpi
    uv run --active examples/convert_paligemma_to_pytorch.py \
      --output_path ../../egomimic/algo/pi_checkpoints/paligemma_base_pytorch
"""

import json
import os
from typing import Literal

import flax.traverse_util
import numpy as np
import safetensors.torch
import torch
import tyro

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.download as download

from convert_jax_model_to_pytorch import slice_paligemma_state_dict


PALIGEMMA_GCS_PATH = "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz"


class _PaliGemmaConfig:
    """Hard-coded PaliGemma 3B architecture constants (matches convert_jax_model_to_pytorch.py)."""

    def __init__(self):
        self.vision_config = type(
            "vc",
            (object,),
            {
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 14,
                "projection_dim": 2048,
            },
        )()
        self.text_config = type(
            "tc",
            (object,),
            {
                "hidden_size": 2048,
                "num_hidden_layers": 18,
                "num_attention_heads": 8,
                "head_dim": 256,
                "intermediate_size": 16384,
            },
        )()


def _load_paligemma_npz(npz_path) -> dict:
    """Load the .npz and return a flat dict keyed like 'img/...', 'llm/...'."""
    with npz_path.open("rb") as f:
        flat = dict(np.load(f, allow_pickle=False))
    nested = flax.traverse_util.unflatten_dict(flat, sep="/")
    return flax.traverse_util.flatten_dict(nested["params"], sep="/")


def main(
    output_path: str,
    precision: Literal["float32", "bfloat16"] = "bfloat16",
    action_dim: int = 32,
    action_horizon: int = 100,
    max_token_len: int = 180,
):
    """Download PaliGemma base weights and emit a PI0Pytorch safetensors checkpoint.

    The PaliGemma backbone is initialized from Google's pretrained weights; the
    action expert and projection heads are left at random init.
    """
    print(f"Fetching {PALIGEMMA_GCS_PATH} ...")
    npz_path = download.maybe_download(PALIGEMMA_GCS_PATH, gs={"token": "anon"})

    print(f"Loading npz from {npz_path} ...")
    paligemma_params = _load_paligemma_npz(npz_path)

    # slice_paligemma_state_dict returns (pytorch_state_dict_for_paligemma, expert_dict).
    # For raw PaliGemma there are no _1 expert keys, so expert_dict should be empty.
    final_state_dict, expert_dict = slice_paligemma_state_dict(paligemma_params, _PaliGemmaConfig())
    if expert_dict:
        print(f"WARNING: unexpected expert keys in raw PaliGemma npz: {list(expert_dict)[:5]}")

    model_config = openpi.models.pi0_config.Pi0Config(
        pi05=True,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)

    missing, unexpected = model.load_state_dict(final_state_dict, strict=False)
    print(f"  Loaded  : {len(final_state_dict)} PaliGemma tensors")
    print(f"  Random  : {len(missing)} tensors (action expert + projection heads)")
    if unexpected:
        print(f"  WARNING : {len(unexpected)} unexpected keys, first 5: {unexpected[:5]}")

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
                "source": PALIGEMMA_GCS_PATH,
            },
            f,
            indent=2,
        )

    print(f"Saved to {output_path}")
    print("Action expert and projection heads are randomly initialized.")


if __name__ == "__main__":
    tyro.cli(main)
