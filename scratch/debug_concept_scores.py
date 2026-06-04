import os
import numpy as np
import torch
from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question

def main():
    checkpoint_path = "/home/aorus/workspaces/intelligent-testing/checkpoints/best-neural-cat-optimized-v14.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LitNeuralCATOptimized.load_from_checkpoint(checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    
    # Giả lập 3 học sinh
    students = [
        {"name": "Giỏi", "theta_true": 1.8},
        {"name": "Trung bình", "theta_true": 0.2},
        {"name": "Yếu", "theta_true": -1.4}
    ]
    
    # Lấy đại diện 30 câu hỏi đầu tiên trong DB để test nhanh
    db_session = SessionLocal()
    db_questions = db_session.query(Question).limit(100).all()
    db_session.close()
    
    q_embeddings = []
    q_features = []
    q_concepts = []
    q_g_priors = []
    q_idxs = []
    
    for idx, q in enumerate(db_questions):
        if q.embedding is None:
            continue
        q_embeddings.append(q.embedding)
        q_concepts.append(q.concept_ids or [0])
        q_idxs.append(idx + 1)
        opt_cnt = q.option_count or 0
        q_g_priors.append(1.0 / opt_cnt if opt_cnt >= 2 else 0.25)
        
        # Tabular features dummy
        q_features.append([0.0] * 22)
        
    q_embeddings_t = torch.tensor(q_embeddings, dtype=torch.float32, device=device)
    q_features_t = torch.tensor(q_features, dtype=torch.float32, device=device)
    q_g_priors_t = torch.tensor(q_g_priors, dtype=torch.float32, device=device)
    q_idxs_t = torch.tensor(q_idxs, dtype=torch.long, device=device)
    
    # Dự đoán độ khó
    with torch.no_grad():
        fused_x = model.model.fusion(q_embeddings_t.unsqueeze(1), q_features_t.unsqueeze(1))
        q_diffs = model.model.predictor.proj_diff(fused_x).squeeze(-1).squeeze(-1).cpu().numpy()
        
    print("Pre-computed diffs:", q_diffs[:10])
    
    for s in students:
        name = s["name"]
        theta_true = s["theta_true"]
        print(f"\n--- Student: {name} (True: {theta_true}) ---")
        
        selected = []
        r_hist = []
        theta_t = 0.0
        
        for step in range(5):
            # Chọn câu
            candidate_indices = [i for i in range(len(q_diffs)) if i not in selected]
            chosen = candidate_indices[np.argmin(np.abs(q_diffs[candidate_indices] - theta_t))]
            selected.append(chosen)
            diff = q_diffs[chosen]
            
            # Làm bài
            P = 0.25 + 0.70 * (1.0 / (1.0 + np.exp(-(theta_true - diff))))
            r = 1.0 if np.random.rand() < P else 0.0
            r_hist.append(r)
            
            # Cập nhật theta_t
            x_emb = torch.zeros(1, 50, 1024, device=device)
            x_feat = torch.zeros(1, 50, 22, device=device)
            r_seq = torch.zeros(1, 50, device=device)
            T_time = torch.zeros(1, 50, device=device)
            concept_indices = torch.full((1, 50, 23), -1, dtype=torch.long, device=device)
            padding_mask = torch.zeros(1, 50, dtype=torch.bool, device=device)
            g_priors = torch.zeros(1, 50, device=device)
            q_idx = torch.zeros(1, 50, dtype=torch.long, device=device)
            
            for t in range(step + 1):
                item = selected[t]
                x_emb[0, t] = q_embeddings_t[item]
                x_feat[0, t] = q_features_t[item]
                r_seq[0, t] = r_hist[t]
                concept_indices[0, t, :len(q_concepts[item])] = torch.tensor(q_concepts[item], dtype=torch.long)
                padding_mask[0, t] = True
                g_priors[0, t] = q_g_priors_t[item]
                q_idx[0, t] = q_idxs_t[item]
                
            with torch.no_grad():
                logits, g, s, se = model(x_emb, x_feat, r_seq, T_time, concept_indices, padding_mask, g_priors, q_indices=q_idx)
                # print theta
                fused_x = model.model.fusion(x_emb, x_feat)
                r_soft, g_r, s_r = model.model.refiner(fused_x, r_seq, g_priors, q_indices=q_idx)
                I = model.model.embedding(fused_x, T_time, r_soft)
                h = model.model.sequence_model(I, padding_mask=padding_mask)
                theta_pred, _ = model.model.decoder(h, concept_indices, padding_mask=padding_mask)
                mastery = model.model.predictor.proj_mastery(theta_pred).squeeze(-1)
                
                theta_t = mastery[0, step].mean().item()
                
            print(f"  Step {step}: Chosen Diff={diff:.2f}, r={int(r)} -> Est Theta={theta_t:.4f}")
            
if __name__ == "__main__":
    main()
