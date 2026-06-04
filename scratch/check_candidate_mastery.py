import os
import numpy as np
import torch
from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question

def scale_to_10(theta):
    # Quy đổi tuyến tính từ [-3.0, 3.0] sang [0.0, 10.0]
    score = ((theta + 3.0) / 6.0) * 10.0
    return np.clip(score, 0.0, 10.0)

def main():
    checkpoint_path = "/home/aorus/workspaces/intelligent-testing/checkpoints/best-neural-cat-optimized-v14.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LitNeuralCATOptimized.load_from_checkpoint(checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    
    # Load 100 câu hỏi từ DB làm item pool
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
        q_features.append([0.0] * 22)
        
    q_embeddings_t = torch.tensor(q_embeddings, dtype=torch.float32, device=device)
    q_features_t = torch.tensor(q_features, dtype=torch.float32, device=device)
    q_g_priors_t = torch.tensor(q_g_priors, dtype=torch.float32, device=device)
    q_idxs_t = torch.tensor(q_idxs, dtype=torch.long, device=device)
    
    with torch.no_grad():
        fused_x = model.model.fusion(q_embeddings_t.unsqueeze(1), q_features_t.unsqueeze(1))
        q_diffs = model.model.predictor.proj_diff(fused_x).squeeze(-1).squeeze(-1).cpu().numpy()
        
    students = [
        {"name": "Học sinh Giỏi", "theta_true": 1.8},
        {"name": "Học sinh Trung bình", "theta_true": 0.2},
        {"name": "Học sinh Yếu", "theta_true": -1.4}
    ]
    
    for s in students:
        name = s["name"]
        theta_true = s["theta_true"]
        
        selected = []
        r_hist = []
        time_hist = []
        theta_t = 0.0
        
        # Mô phỏng 30 câu hỏi
        for step in range(30):
            candidate_indices = [i for i in range(len(q_diffs)) if i not in selected]
            chosen = candidate_indices[np.argmin(np.abs(q_diffs[candidate_indices] - theta_t))]
            selected.append(chosen)
            diff = q_diffs[chosen]
            
            P = 0.25 + 0.70 * (1.0 / (1.0 + np.exp(-(theta_true - diff))))
            r = 1.0 if np.random.rand() < P else 0.0
            r_hist.append(r)
            time_hist.append(20.0)
            
            # Ước lượng theta_t tạm thời
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
                T_time[0, t] = time_hist[t]
                concept_indices[0, t, :len(q_concepts[item])] = torch.tensor(q_concepts[item], dtype=torch.long)
                padding_mask[0, t] = True
                g_priors[0, t] = q_g_priors_t[item]
                q_idx[0, t] = q_idxs_t[item]
                
            with torch.no_grad():
                fused_x_seq = model.model.fusion(x_emb, x_feat)
                r_soft, g_r, s_r = model.model.refiner(fused_x_seq, r_seq, g_priors, q_indices=q_idx)
                I = model.model.embedding(fused_x_seq, T_time, r_soft)
                h = model.model.sequence_model(I, padding_mask=padding_mask)
                theta_pred, _ = model.model.decoder(h, concept_indices, padding_mask=padding_mask)
                mastery = model.model.predictor.proj_mastery(theta_pred).squeeze(-1)
                theta_t = mastery[0, step].mean().item()
                
        # Sau 30 câu hỏi, lấy context cuối cùng và tính mastery cho TOÀN BỘ concept
        with torch.no_grad():
            fused_x_seq = model.model.fusion(x_emb, x_feat)
            r_soft, g_r, s_r = model.model.refiner(fused_x_seq, r_seq, g_priors, q_indices=q_idx)
            I = model.model.embedding(fused_x_seq, T_time, r_soft)
            h = model.model.sequence_model(I, padding_mask=padding_mask)
            
            h_proj = model.model.decoder.proj_h(h)[:, 29, :] # shape (1, d_h)
            
            # Tính mastery cho tất cả concept từ 0 đến model.model.decoder.K (loại trừ dummy K)
            K = model.model.decoder.K
            all_c_indices = torch.arange(K, device=device) # shape (K,)
            c_embs = model.model.decoder.concept_embedding(all_c_indices) # shape (K, d_h)
            
            h_proj_expanded = h_proj.expand(K, -1) # shape (K, d_h)
            fused_input = torch.cat([h_proj_expanded, c_embs], dim=-1) # shape (K, 2 * d_h)
            theta_cands = model.model.decoder.candidate_generator(fused_input) # shape (K, d_h)
            all_masteries = model.model.predictor.proj_mastery(theta_cands).squeeze(-1).cpu().numpy()
            
            # Chọn ra 5 concept có trong database để hiển thị
            # Lấy các concept thực sự xuất hiện trong selected items của học sinh này
            active_concepts = []
            for item in selected:
                for c in q_concepts[item]:
                    if c != -1 and c != 0 and c not in active_concepts:
                        active_concepts.append(c)
            
            test_concepts = active_concepts[:5]
            while len(test_concepts) < 5:
                test_concepts.append(len(test_concepts) + 1)
                
            print(f"\nStudent: {name} (True: {theta_true})")
            print(f"  r_hist (sum={sum(r_hist)}): {r_hist}")
            print(f"  All Concepts Mastery Stats: Min={all_masteries.min():.4f}, Max={all_masteries.max():.4f}, Mean={all_masteries.mean():.4f}, Std={all_masteries.std():.4f}")
            
            concept_masteries = {c: all_masteries[c] for c in test_concepts}
            overall_theta = np.mean(list(concept_masteries.values()))
            print(f"  Overall Theta (selected concepts): {overall_theta:.4f} -> Scale 10: {scale_to_10(overall_theta):.2f}")
            for c, m in concept_masteries.items():
                print(f"    Concept {c}: Theta={m:.4f} -> Scale 10: {scale_to_10(m):.2f}")

if __name__ == "__main__":
    main()

