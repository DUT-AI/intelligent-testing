import torch
from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized

checkpoint_path = "/home/aorus/workspaces/intelligent-testing/checkpoints/best-neural-cat-optimized-v14.ckpt"
print("Loading model...")
model = LitNeuralCATOptimized.load_from_checkpoint(checkpoint_path, strict=False)
model.cpu()
model.eval()

# Tạo tensor giả lập
B, T = 4, 10
d_x = 1024
d_feat = 22
max_c = 23

x_emb = torch.randn(B, T, d_x)
x_feat = torch.randn(B, T, d_feat)
r = torch.randint(0, 2, (B, T)).float()
T_time = torch.rand(B, T) * 30.0
concept_indices = torch.randint(0, 100, (B, T, max_c))
padding_mask = torch.ones(B, T, dtype=torch.bool)
g_priors = torch.full((B, T), 0.25)

with torch.no_grad():
    # 1. Dự đoán trực tiếp qua model
    logits, g, s, se = model(
        x_emb, x_feat, r, T_time, concept_indices, padding_mask, g_priors
    )
    
    # 2. Xem các giá trị g và s
    print("\n--- Model Outputs ---")
    print(f"g: min={g.min().item():.8f}, max={g.max().item():.8f}, mean={g.mean().item():.8f}")
    print(f"s: min={s.min().item():.8f}, max={s.max().item():.8f}, mean={s.mean().item():.8f}")
    print(f"logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
    print(f"se: min={se.min().item():.4f}, max={se.max().item():.4f}")
