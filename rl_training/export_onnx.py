#!/usr/bin/env python3
"""Export trained Microduck PPO Policy to ONNX.

This script extracts the actor network from Stable-Baselines3 PPO
and exports it as an ONNX model adhering to the Pollen Robotics robotd contract:
  - Input:  'obs' of shape [1, 61] (float32)
  - Output: 'actions' of shape [1, 14] (float32)
"""

from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO


class OnnxActorWrapper(nn.Module):
    """Wrapper that extracts deterministic action predictions for ONNX export."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.policy.extract_features(obs)
        # Actor latent representation
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        # Action mean (deterministic output)
        action_mean = self.policy.action_net(latent_pi)
        return action_mean


def parse_args():
    parser = argparse.ArgumentParser(description="Export Microduck Policy to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/microduck_final.zip",
        help="Path to trained SB3 PPO model zip",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="exported/microduck_policy.onnx",
        help="Path where ONNX file will be saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model checkpoint not found at: {args.model_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print(f"Loading checkpoint from: {args.model_path}")
    model = PPO.load(args.model_path, device="cpu")
    actor = OnnxActorWrapper(model.policy)
    actor.eval()

    dummy_obs = torch.zeros((1, 61), dtype=torch.float32)

    with torch.no_grad():
        test_out = actor(dummy_obs)
        print(f"Test inference output shape: {test_out.shape}")
        assert test_out.shape == (1, 14), f"Expected (1, 14), got {test_out.shape}"

    print(f"Exporting ONNX model to: {args.output_path}...")
    torch.onnx.export(
        actor,
        dummy_obs,
        args.output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
    )

    print("=" * 65)
    print("✅ ONNX Export Successful!")
    print(f"Saved to: {args.output_path}")
    print("Model Input:  'obs'     -> [batch_size, 61]")
    print("Model Output: 'actions' -> [batch_size, 14]")
    print("=" * 65)


if __name__ == "__main__":
    main()
