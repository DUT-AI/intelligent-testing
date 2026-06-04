import os
import numpy as np
import torch
import torch.nn.functional as F

from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question

def main():
    checkpoint_path = "/home/aorus/workspaces/intelligent-testing/checkpoints/best-neural-cat-optimized-v14.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load model
    print(f"Loading model from checkpoint: {checkpoint_path} ...")
    model = LitNeuralCATOptimized.load_from_checkpoint(checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    
    # 2. Load questions
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
    
    # 3. Simulated Students
    simulated_students = [
        {"name": "Học sinh Giỏi", "theta_true": 1.8},
        {"name": "Học sinh Trung bình", "theta_true": 0.1},
        {"name": "Học sinh Yếu", "theta_true": -1.4}
    ]
    
    num_steps = 30
    max_seq_len = 50
    max_c = 23
    
    results = {}
    
    for student in simulated_students:
        name = student["name"]
        theta_true = student["theta_true"]
        print(f"Simulating {name}...")
        
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
            
            T_time_val = np.random.exponential(15.0) + 5.0 if r_t == 1.0 else np.random.exponential(25.0) + 5.0
            T_time_val = np.clip(T_time_val, 2.0, 120.0)
            time_history.append(T_time_val)
            
            # Cập nhật theta_t: trung bình của các concept đã làm từ đầu đến bước hiện tại
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
                theta_pred, log_var = model.model.decoder(h, concept_indices_seq, padding_mask=padding_mask_seq)
                
                mastery = model.model.predictor.proj_mastery(theta_pred).squeeze(-1)
                # Tối ưu: Lấy trung bình năng lực của toàn bộ chuỗi lịch sử đã làm để phản ánh đúng năng lực hiện tại
                theta_t = mastery[0, :step+1].mean().item()

        # Thu thập các active concepts thực sự xuất hiện trong bài thi của học sinh này
        active_concepts_in_session = []
        concept_correct_counts = {}
        concept_total_counts = {}
        
        for t, idx in enumerate(selected_item_indices):
            for c in q_concepts[idx]:
                if c != -1 and c != 0:
                    if c not in active_concepts_in_session:
                        active_concepts_in_session.append(c)
                    concept_total_counts[c] = concept_total_counts.get(c, 0) + 1
                    if r_history[t] == 1.0:
                        concept_correct_counts[c] = concept_correct_counts.get(c, 0) + 1
        
        # Chọn ra 5 active concepts đầu tiên để chẩn đoán
        target_concepts = active_concepts_in_session[:5]
        while len(target_concepts) < 5:
            target_concepts.append(len(target_concepts) + 1)
            
        # 3.2 Tính năng lực dự kiến (Candidate Mastery) cho các concept mục tiêu dựa trên context cuối cùng
        with torch.no_grad():
            fused_x_seq = model.model.fusion(x_emb_seq, x_feat_seq)
            r_soft, g, s = model.model.refiner(fused_x_seq, r_seq, g_priors_seq, q_indices=q_idx_seq)
            I = model.model.embedding(fused_x_seq, T_time_seq, r_soft)
            h = model.model.sequence_model(I, padding_mask=padding_mask_seq)
            
            # Lấy context cuối cùng tại bước 29 (sau 30 câu hỏi)
            h_proj = model.model.decoder.proj_h(h)[:, 29, :]  # shape (1, d_h)
            
            final_masteries = {}
            for c in target_concepts:
                c_idx_t = torch.tensor([c], device=device)
                c_emb = model.model.decoder.concept_embedding(c_idx_t)  # shape (1, d_h)
                
                # Nối context với concept embedding
                fused_input = torch.cat([h_proj, c_emb], dim=-1)  # shape (1, 2 * d_h)
                theta_cand = model.model.decoder.candidate_generator(fused_input)  # shape (1, d_h)
                mastery_c = model.model.predictor.proj_mastery(theta_cand).item()
                
                # Điều chỉnh nhẹ theo tỉ lệ làm đúng thực tế của concept đó nếu có làm
                if c in concept_total_counts:
                    ratio = concept_correct_counts.get(c, 0) / concept_total_counts[c]
                    # Kết hợp: 70% từ candidate mastery của mô hình (đã qua Transformer context) + 30% kết quả trực tiếp
                    # Điều này giúp điểm số vừa giữ được đặc tính suy diễn thông minh vừa tôn trọng kết quả thực tế
                    mastery_c = 0.7 * mastery_c + 0.3 * (ratio * 4.0 - 2.0) # mapping ratio [0, 1] -> [-2, 2]
                
                final_masteries[c] = mastery_c
                
            overall_theta = np.mean(list(final_masteries.values()))

        results[name] = {
            'final_theta': overall_theta,
            'concept_masteries': final_masteries,
            'concepts_tested': target_concepts,
            'correct_count': sum(r_history)
        }
        
    # Quy đổi điểm số sang thang 10 Việt Nam:
    # Điểm chung: quy đổi tuyến tính từ tỉ lệ câu đúng của học sinh, kết hợp với trọng số từ Theta ước lượng của mô hình
    # để vừa phản ánh độ phân hóa tuyệt đối (Giỏi > Trung bình > Yếu) vừa mang tính khoa học của CAT.
    # Điểm từng concept: quy đổi dựa trên candidate mastery của concept đó.
    
    # Định nghĩa dải Theta chuẩn cho concept: [-2.5, 2.5] -> [0.0, 10.0]
    def scale_concept_theta(theta):
        score = ((theta + 2.5) / 5.0) * 10.0
        return np.clip(score, 0.0, 10.0)

    # In bảng kết quả
    print("\n" + "="*120)
    print("                                BẢNG ĐIỂM QUY ĐỔI SANG THANG ĐIỂM 10")
    print("="*120)
    print(f"| {'Học sinh':<22} | {'Số câu đúng':<12} | {'Điểm chung (Thang 10)':<22} | {'Concept 1':<12} | {'Concept 2':<12} | {'Concept 3':<12} | {'Concept 4':<12} | {'Concept 5':<12} |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name, res in results.items():
        correct_cnt_int = int(res['correct_count'])
        
        # Tính điểm chung kết hợp: 70% từ tỉ lệ đúng thực tế (đảm bảo phân hóa rõ rệt Giỏi > TB > Yếu) + 30% từ Theta của CAT
        ratio_score = (correct_cnt_int / num_steps) * 10.0
        
        # Ánh xạ Theta chung của học sinh sang thang 10 (giả định Theta nằm trong [-2.0, 2.0])
        theta_score = ((res['final_theta'] + 2.0) / 4.0) * 10.0
        theta_score = np.clip(theta_score, 0.0, 10.0)
        
        overall_score = 0.7 * ratio_score + 0.3 * theta_score
        
        concept_scores = []
        for c in res['concepts_tested']:
            c_theta = res['concept_masteries'][c]
            c_score = scale_concept_theta(c_theta)
            concept_scores.append(f"C{c}: {c_score:.2f}")
            
        concept_str = " | ".join(concept_scores)
        print(f"| {name:<22} | {correct_cnt_int:^12d} | **{overall_score:.2f}** (Theta: {res['final_theta']:+.3f}) | {concept_str} |")
    print("="*120)

if __name__ == "__main__":
    main()
