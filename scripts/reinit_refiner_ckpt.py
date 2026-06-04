import torch
import torch.nn as nn
import math

ckpt_path = "/home/aorus/workspaces/intelligent-testing/checkpoints/best-neural-cat-optimized-v14.ckpt"
out_ckpt_path = "/home/aorus/workspaces/intelligent-testing/checkpoints/best-neural-cat-optimized-v14-reinit.ckpt"

print(f"Loading checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint["state_dict"]

# 1. Re-initialize mlp_item.0 (Linear from d_x=1024 to 64)
w0 = state_dict["model.refiner.mlp_item.0.weight"]
b0 = state_dict["model.refiner.mlp_item.0.bias"]
nn.init.kaiming_uniform_(w0, a=math.sqrt(5))
fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w0)
bound0 = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
nn.init.uniform_(b0, -bound0, bound0)
print(f"Re-initialized mlp_item.0: weight min={w0.min():.4f}, max={w0.max():.4f}; bias mean={b0.mean():.4f}")

# 2. Re-initialize mlp_item.2 (Linear from 64 to 2)
w2 = state_dict["model.refiner.mlp_item.2.weight"]
b2 = state_dict["model.refiner.mlp_item.2.bias"]
nn.init.kaiming_uniform_(w2, a=math.sqrt(5))
fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(w2)
bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
nn.init.uniform_(b2, -bound2, bound2)
print(f"Re-initialized mlp_item.2: weight min={w2.min():.4f}, max={w2.max():.4f}; bias mean={b2.mean():.4f}")

# 3. Reset question-specific biases to 0
if "model.refiner.q_g_bias.weight" in state_dict:
    q_g = state_dict["model.refiner.q_g_bias.weight"]
    nn.init.zeros_(q_g)
    print("Reset q_g_bias to zeros.")
if "model.refiner.q_s_bias.weight" in state_dict:
    q_s = state_dict["model.refiner.q_s_bias.weight"]
    nn.init.zeros_(q_s)
    print("Reset q_s_bias to zeros.")

# Save the new checkpoint
torch.save(checkpoint, out_ckpt_path)
print(f"Successfully saved re-initialized checkpoint to {out_ckpt_path}")
