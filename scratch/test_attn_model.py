import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.neural_cat_attn import NeuralCATEngineAttn

def test_attn_model():
    print("Initializing NeuralCATEngineAttn...")
    B, T = 4, 80
    d_embedding = 1024
    d_features = 22
    d_time = 32
    d_h = 128
    K = 10
    nhead = 4
    num_layers = 2
    max_c = 5
    num_questions = 100

    model = NeuralCATEngineAttn(
        d_embedding=d_embedding,
        d_features=d_features,
        d_time=d_time,
        d_h=d_h,
        K=K,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=200,
        num_questions=num_questions,
    )

    print("Generating random inputs...")
    x_emb = torch.randn(B, T, d_embedding)
    x_feat = torch.randn(B, T, d_features)
    r = torch.randint(0, 2, (B, T)).float()
    T_time = torch.randn(B, T).abs() * 10.0
    
    # Concept indices: values between 0 and K-1, or -1 for padding
    concept_indices = torch.randint(-1, K, (B, T, max_c))
    
    padding_mask = torch.ones(B, T, dtype=torch.bool)
    # Set some elements to False to mock padding
    padding_mask[:, T-10:] = False
    
    g_priors = torch.full((B, T), 0.25)
    q_indices = torch.randint(0, num_questions, (B, T))

    print("Running forward pass...")
    output = model(
        x_emb=x_emb,
        x_feat=x_feat,
        r=r,
        T_time=T_time,
        concept_indices=concept_indices,
        padding_mask=padding_mask,
        g_priors=g_priors,
        q_indices=q_indices,
    )

    print("Checking output shapes...")
    print(f"Logits shape: {output.logits.shape}")
    print(f"Guessing (g) shape: {output.g.shape}")
    print(f"Slip (s) shape: {output.s.shape}")
    print(f"Standard Error (se) shape: {output.se.shape}")

    assert output.logits.shape == (B, T)
    assert output.g.shape == (B, T)
    assert output.s.shape == (B, T)
    assert output.se.shape == (B, T)

    print("Running backward pass...")
    loss = output.logits.sum()
    loss.backward()
    print("Backward pass completed successfully!")
    print("All checks passed!")

if __name__ == "__main__":
    test_attn_model()
