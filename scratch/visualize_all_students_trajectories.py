import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question

def main():
    checkpoint_path = "/home/aorus/workspaces/intelligent-testing/checkpoints/best-neural-cat-optimized-v14.ckpt"
    artifacts_dir = "/home/aorus/.gemini/antigravity-ide/brain/180e48e2-5aa8-4e26-9249-cb1014c01996"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Load model
    print(f"Loading model from checkpoint: {checkpoint_path} ...")
    model = LitNeuralCATOptimized.load_from_checkpoint(checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    
    # 2. Load questions from DB
    print("Loading questions from database...")
    db_session = SessionLocal()
    try:
        from sqlalchemy.orm import joinedload
        db_questions = (
            db_session.query(Question)
            .options(joinedload(Question.features), joinedload(Question.misconceptions))
            .all()
        )
        print(f"Loaded {len(db_questions)} questions.")
    finally:
        db_session.close()
        
    q_ids = []
    q_idxs = []
    q_embeddings = []
    q_features = []
    q_concepts = []
    q_g_priors = []
    
    for idx, q in enumerate(db_questions):
        if q.embedding is None:
            continue
        q_ids.append(q.id)
        q_idxs.append(idx + 1)
        q_embeddings.append(q.embedding)
        q_concepts.append(q.concept_ids or [0])
        
        feat_vec = []
        def safe_float(v):
            return float(v) if v is not None else 0.0
            
        if q.features is not None:
            feat_vec.extend([
                safe_float(q.features.word_count), safe_float(q.features.avg_word_length),
                safe_float(q.features.avg_sentence_length), safe_float(q.features.vocab_difficulty),
                safe_float(q.features.syntactic_complexity), safe_float(q.features.p_concrete),
                safe_float(q.features.p_symbol), safe_float(q.features.p_abstract),
                safe_float(q.features.inference_steps), safe_float(q.features.q1_tinhtoan),
                safe_float(q.features.q2_lythuyetso), safe_float(q.features.q3_hinhhoc),
                safe_float(q.features.q4_chuyendong), safe_float(q.features.q5_toandokinhdien),
                safe_float(q.features.q6_tonghieuti), safe_float(q.features.q7_dem_tohop),
                safe_float(q.features.q8_logic_trochoi)
            ])
        else:
            feat_vec.extend([0.0] * 17)
            
        if q.misconceptions is not None:
            feat_vec.extend([
                safe_float(q.misconceptions.llm_arithmetic), safe_float(q.misconceptions.llm_procedural),
                safe_float(q.misconceptions.llm_conceptual), safe_float(q.misconceptions.llm_lack_of_sense),
                safe_float(q.misconceptions.llm_misconception_score)
            ])
        else:
            feat_vec.extend([0.0] * 5)
            
        q_features.append(feat_vec)
        
        opt_cnt = q.option_count or 0
        q_g_priors.append(1.0 / opt_cnt if opt_cnt >= 2 else 0.25)

    q_embeddings_t = torch.tensor(q_embeddings, dtype=torch.float32, device=device)
    q_features_t = torch.tensor(q_features, dtype=torch.float32, device=device)
    q_g_priors_t = torch.tensor(q_g_priors, dtype=torch.float32, device=device)
    q_idxs_t = torch.tensor(q_idxs, dtype=torch.long, device=device)
    
    pool_size = len(q_ids)
    q_diffs = []
    q_gs = []
    q_ss = []
    
    with torch.no_grad():
        chunk_size = 512
        for i in range(0, pool_size, chunk_size):
            end_idx = min(i + chunk_size, pool_size)
            x_emb = q_embeddings_t[i:end_idx].unsqueeze(1)
            x_feat = q_features_t[i:end_idx].unsqueeze(1)
            g_priors = q_g_priors_t[i:end_idx].unsqueeze(1)
            q_idx = q_idxs_t[i:end_idx].unsqueeze(1)
            
            fused_x = model.model.fusion(x_emb, x_feat)
            _, g, s = model.model.refiner(fused_x, torch.zeros_like(g_priors), g_priors, q_indices=q_idx)
            
            diff = model.model.predictor.proj_diff(fused_x).squeeze(-1)
            if model.model.predictor.q_diff_bias is not None:
                safe_q_indices = torch.where(
                    (q_idx >= 0) & (q_idx < model.model.predictor.q_diff_bias.num_embeddings),
                    q_idx,
                    0
                )
                diff = diff + model.model.predictor.q_diff_bias(safe_q_indices).squeeze(-1)
                
            q_diffs.extend(diff.squeeze(-1).cpu().numpy())
            q_gs.extend(g.squeeze(-1).cpu().numpy())
            q_ss.extend(s.squeeze(-1).cpu().numpy())
            
    q_diffs = np.array(q_diffs)
    q_gs = np.array(q_gs)
    q_ss = np.array(q_ss)
    
    # 3. Định nghĩa 3 đối tượng học sinh
    simulated_students = [
        {"name": "Học sinh Giỏi", "theta_true": 1.8, "file_name": "student_gioi_trajectory.png"},
        {"name": "Học sinh Trung bình", "theta_true": 0.2, "file_name": "student_trung_binh_trajectory.png"},
        {"name": "Học sinh Yếu", "theta_true": -1.4, "file_name": "student_yeu_trajectory.png"}
    ]
    
    num_steps = 30
    max_seq_len = 50
    max_c = 23
    K = model.model.decoder.K
    
    # Định nghĩa dải Theta chuẩn để scale sang thang 10: [-2.0, 2.0] -> [0.0, 10.0]
    def scale_theta_to_10(theta):
        score = ((theta + 2.0) / 4.0) * 10.0
        return np.clip(score, 0.0, 10.0)

    for student in simulated_students:
        name = student["name"]
        theta_true = student["theta_true"]
        file_name = student["file_name"]
        print(f"\nSimulating {name} (True Theta = {theta_true:.2f}) ...")
        
        selected_item_indices = []
        r_history = []
        time_history = []
        
        # 3.1 Chạy mô phỏng CAT thích ứng 30 bước
        theta_t = 0.0
        for step in range(num_steps):
            candidate_indices = [i for i in range(pool_size) if i not in selected_item_indices]
            diffs = q_diffs[candidate_indices]
            best_cand_idx = np.argmin(np.abs(diffs - theta_t))
            chosen_item_idx = candidate_indices[best_cand_idx]
            
            selected_item_indices.append(chosen_item_idx)
            chosen_diff = q_diffs[chosen_item_idx]
            
            g_val = q_gs[chosen_item_idx]
            s_val = q_ss[chosen_item_idx]
            
            P_correct = g_val + (1.0 - s_val - g_val) * (1.0 / (1.0 + np.exp(-(theta_true - chosen_diff))))
            r_t = 1.0 if np.random.rand() < P_correct else 0.0
            r_history.append(r_t)
            
            time_history.append(20.0)
            
            # Cập nhật theta_t tạm thời
            x_emb_seq = torch.zeros(1, max_seq_len, 1024, device=device)
            x_feat_seq = torch.zeros(1, max_seq_len, 22, device=device)
            r_seq = torch.zeros(1, max_seq_len, device=device)
            T_time_seq = torch.zeros(1, max_seq_len, device=device)
            concept_indices_seq = torch.full((1, max_seq_len, max_c), -1, dtype=torch.long, device=device)
            padding_mask_seq = torch.zeros(1, max_seq_len, dtype=torch.bool, device=device)
            g_priors_seq = torch.zeros(1, max_seq_len, device=device)
            q_idx_seq = torch.zeros(1, max_seq_len, dtype=torch.long, device=device)
            
            for t in range(step + 1):
                item_idx = selected_item_indices[t]
                x_emb_seq[0, t] = q_embeddings_t[item_idx]
                x_feat_seq[0, t] = q_features_t[item_idx]
                r_seq[0, t] = r_history[t]
                T_time_seq[0, t] = time_history[t]
                c_ids = q_concepts[item_idx]
                concept_indices_seq[0, t, :len(c_ids)] = torch.tensor(c_ids, dtype=torch.long)
                padding_mask_seq[0, t] = True
                g_priors_seq[0, t] = q_g_priors_t[item_idx]
                q_idx_seq[0, t] = q_idxs_t[item_idx]
                
            with torch.no_grad():
                fused_x = model.model.fusion(x_emb_seq, x_feat_seq)
                r_soft, g, s = model.model.refiner(fused_x, r_seq, g_priors_seq, q_indices=q_idx_seq)
                I = model.model.embedding(fused_x, T_time_seq, r_soft)
                h = model.model.sequence_model(I, padding_mask=padding_mask_seq)
                theta_pred, _ = model.model.decoder(h, concept_indices_seq, padding_mask=padding_mask_seq)
                mastery = model.model.predictor.proj_mastery(theta_pred).squeeze(-1)
                theta_t = mastery[0, :step+1].mean().item()
                
        # 3.2 Thu thập TẤT CẢ các concept đã xuất hiện trong bài thi
        all_tested_concepts = set()
        concept_occurrences = {} # c -> list of (step, r_t)
        
        for t, item_idx in enumerate(selected_item_indices):
            c_ids = q_concepts[item_idx]
            for c in c_ids:
                if c != -1 and c != 0:
                    all_tested_concepts.add(c)
                    if c not in concept_occurrences:
                        concept_occurrences[c] = []
                    concept_occurrences[c].append((t, r_history[t]))
                    
        # Sắp xếp các concept theo tần suất xuất hiện giảm dần và chỉ lấy top 5
        sorted_active_concepts = sorted(concept_occurrences.keys(), key=lambda c: len(concept_occurrences[c]), reverse=True)
        tested_concepts_list = sorted_active_concepts[:5]
        
        # Chọn concept đối chứng (control) không xuất hiện trong bài thi
        control_c = None
        for c in range(1, K):
            if c not in all_tested_concepts:
                control_c = c
                break
        if control_c is None:
            control_c = 999
            
        print(f"  - {name} correct_count: {sum(r_history)}/30")
        print(f"  - Total active concepts tested: {len(tested_concepts_list)} ({tested_concepts_list})")
        print(f"  - Concept {control_c} selected as Control (never tested).")
        
        # 3.3 Tính toán trajectory cho TẤT CẢ các concept hoạt động và đối chứng
        all_track_concepts = tested_concepts_list + [control_c]
        trajectory_data = {c: [] for c in all_track_concepts}
        
        for step in range(num_steps):
            x_emb_seq = torch.zeros(1, max_seq_len, 1024, device=device)
            x_feat_seq = torch.zeros(1, max_seq_len, 22, device=device)
            r_seq = torch.zeros(1, max_seq_len, device=device)
            T_time_seq = torch.zeros(1, max_seq_len, device=device)
            concept_indices_seq = torch.full((1, max_seq_len, max_c), -1, dtype=torch.long, device=device)
            padding_mask_seq = torch.zeros(1, max_seq_len, dtype=torch.bool, device=device)
            g_priors_seq = torch.zeros(1, max_seq_len, device=device)
            q_idx_seq = torch.zeros(1, max_seq_len, dtype=torch.long, device=device)
            
            for t in range(step + 1):
                item_idx = selected_item_indices[t]
                x_emb_seq[0, t] = q_embeddings_t[item_idx]
                x_feat_seq[0, t] = q_features_t[item_idx]
                r_seq[0, t] = r_history[t]
                T_time_seq[0, t] = time_history[t]
                c_ids = q_concepts[item_idx]
                concept_indices_seq[0, t, :len(c_ids)] = torch.tensor(c_ids, dtype=torch.long)
                padding_mask_seq[0, t] = True
                g_priors_seq[0, t] = q_g_priors_t[item_idx]
                q_idx_seq[0, t] = q_idxs_t[item_idx]
                
            with torch.no_grad():
                fused_x = model.model.fusion(x_emb_seq, x_feat_seq)
                r_soft, g, s = model.model.refiner(fused_x, r_seq, g_priors_seq, q_indices=q_idx_seq)
                I = model.model.embedding(fused_x, T_time_seq, r_soft)
                h = model.model.sequence_model(I, padding_mask=padding_mask_seq)
                
                h_projected = model.model.decoder.proj_h(h)
                alpha = model.model.decoder._get_damping_schedule(max_seq_len, device)
                
                theta_prev = model.model.decoder.theta_0.unsqueeze(0)
                for t in range(step + 1):
                    c_indices_t = concept_indices_seq[:, t]
                    valid_mask_t = c_indices_t != -1
                    safe_indices_t = torch.where(valid_mask_t, c_indices_t, model.model.decoder.K)
                    safe_indices_expanded = safe_indices_t.unsqueeze(-1).expand(-1, -1, model.model.decoder.d_h)
                    
                    theta_selected_t = torch.gather(theta_prev, dim=1, index=safe_indices_expanded)
                    
                    c_emb = model.model.decoder.concept_embedding(safe_indices_t)
                    h_proj_t = h_projected[:, t, :]
                    h_expanded = h_proj_t.unsqueeze(1).expand(-1, max_c, -1)
                    fused_input = torch.cat([h_expanded, c_emb], dim=-1)
                    theta_cand_t = model.model.decoder.candidate_generator(fused_input)
                    
                    alpha_t = alpha[t]
                    mask_t = padding_mask_seq[:, t].view(-1, 1, 1).float()
                    decay_t = alpha_t * valid_mask_t.unsqueeze(-1).float() * mask_t
                    
                    theta_update = torch.where(
                        decay_t > 0,
                        (1.0 - decay_t) * theta_selected_t + decay_t * theta_cand_t,
                        theta_selected_t,
                    )
                    theta_prev = theta_prev.scatter(dim=1, index=safe_indices_expanded, src=theta_update)
                    
                for c in all_track_concepts:
                    theta_c = theta_prev[0, c, :]
                    mastery_c = model.model.predictor.proj_mastery(theta_c.unsqueeze(0)).item()
                    
                    score_c = scale_theta_to_10(mastery_c)
                    trajectory_data[c].append(score_c)
                    
        # 3.4 Vẽ biểu đồ cho học sinh này
        plt.figure(figsize=(13, 8))
        steps = np.arange(1, num_steps + 1)
        
        # Tạo bảng màu đa dạng dựa trên số lượng concept thi
        colormap = plt.cm.tab20
        num_colors = len(tested_concepts_list)
        colors = [colormap(i / max(1, num_colors - 1)) for i in range(num_colors)]
        
        # Vẽ các concept đã thi
        for idx_c, c in enumerate(tested_concepts_list):
            plt.plot(steps, trajectory_data[c], label=f"Concept {c}", color=colors[idx_c], linewidth=2.0)
            
            # Đánh dấu các mốc làm đúng (tròn xanh) và làm sai (X đỏ) của concept này
            occs = concept_occurrences[c]
            
            correct_steps = [occ[0] + 1 for occ in occs if occ[1] == 1.0]
            correct_scores = [trajectory_data[c][s - 1] for s in correct_steps]
            if correct_steps:
                plt.scatter(correct_steps, correct_scores, color="green", marker="o", s=80, edgecolors="black", zorder=5)
                
            incorrect_steps = [occ[0] + 1 for occ in occs if occ[1] == 0.0]
            incorrect_scores = [trajectory_data[c][s - 1] for s in incorrect_steps]
            if incorrect_steps:
                plt.scatter(incorrect_steps, incorrect_scores, color="red", marker="X", s=80, edgecolors="black", zorder=5)

        # Vẽ concept đối chứng
        plt.plot(steps, trajectory_data[control_c], label=f"Concept {control_c} (Đối chứng - Không thi)", color="black", linestyle="--", linewidth=2.5)
        
        plt.title(f"Lịch Sử Năng Lực Concept Tích Lũy - {name} (Đúng: {sum(r_history)}/30)", fontsize=13, fontweight="bold")
        plt.xlabel("Số câu hỏi đã trả lời (Step t)", fontsize=11)
        plt.ylabel("Điểm số Concept (Thang điểm 10)", fontsize=11)
        plt.ylim(0.0, 10.5)
        plt.xlim(0.5, 30.5)
        plt.xticks(np.arange(1, 31, 1))
        plt.grid(True, linestyle=":", alpha=0.5)
        
        # Legend hiển thị cột đôi nếu quá nhiều concept
        ncol = 2 if len(all_track_concepts) > 8 else 1
        plt.legend(loc="upper left", fontsize=9, ncol=ncol)
        
        plt.tight_layout()
        plot_path = os.path.join(artifacts_dir, file_name)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot for {name} to {plot_path}")
        
    print("\nSimulating and plotting trajectories for all students completed successfully!")

if __name__ == "__main__":
    main()
