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
    
    # 2. Load all questions from database to form the Item Pool
    print("Loading question metadata from database...")
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
        
    # 3. Pre-compute item parameters for the entire pool
    print("Pre-computing item parameters (difficulty, guessing, slip)...")
    q_ids = []
    q_embeddings = []
    q_features = []
    q_concepts = []
    q_g_priors = []
    q_idxs = []  # integer indices
    
    for idx, q in enumerate(db_questions):
        if q.embedding is None:
            continue
        q_ids.append(q.id)
        q_idxs.append(idx + 1)  # 1-indexed to match model training questions
        q_embeddings.append(q.embedding)
        q_concepts.append(q.concept_ids or [0])
        
        # Tabular features
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

    # Convert to Tensors
    q_embeddings_t = torch.tensor(q_embeddings, dtype=torch.float32, device=device)
    q_features_t = torch.tensor(q_features, dtype=torch.float32, device=device)
    q_g_priors_t = torch.tensor(q_g_priors, dtype=torch.float32, device=device)
    q_idxs_t = torch.tensor(q_idxs, dtype=torch.long, device=device)
    
    # Batch predict difficulty, guessing, slip
    pool_size = len(q_ids)
    q_diffs = []
    q_gs = []
    q_ss = []
    
    with torch.no_grad():
        # Predict parameters in chunks to avoid GPU OOM
        chunk_size = 512
        for i in range(0, pool_size, chunk_size):
            end_idx = min(i + chunk_size, pool_size)
            x_emb = q_embeddings_t[i:end_idx].unsqueeze(1)  # (B, 1, d_x)
            x_feat = q_features_t[i:end_idx].unsqueeze(1)  # (B, 1, d_feat)
            g_priors = q_g_priors_t[i:end_idx].unsqueeze(1)  # (B, 1)
            q_idx = q_idxs_t[i:end_idx].unsqueeze(1)  # (B, 1)
            
            fused_x = model.model.fusion(x_emb, x_feat)
            _, g, s = model.model.refiner(fused_x, torch.zeros_like(g_priors), g_priors, q_indices=q_idx)
            
            diff = model.model.predictor.proj_diff(fused_x).squeeze(-1)
            if model.model.predictor.q_diff_bias is not None:
                # Map out-of-bound indices safely
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
    
    print("Done pre-computing parameters.")
    
    # 4. Set up Online Simulation
    # 3 simulated students
    simulated_students = [
        {"name": "Học sinh Giỏi", "theta_true": 1.5, "color": "green", "marker": "o"},
        {"name": "Học sinh Trung bình", "theta_true": 0.0, "color": "blue", "marker": "s"},
        {"name": "Học sinh Yếu", "theta_true": -1.2, "color": "red", "marker": "^"}
    ]
    
    num_steps = 30
    max_seq_len = 50
    max_c = 23
    
    results = {}
    
    for student in simulated_students:
        name = student["name"]
        theta_true = student["theta_true"]
        print(f"\nSimulating {name} (True Theta = {theta_true:.2f}) ...")
        
        # Lịch sử tương tác lưu index trong pool
        selected_item_indices = []
        r_history = []
        time_history = []
        
        theta_history = []
        diff_history = []
        se_history = []
        
        for step in range(num_steps):
            # 4.1. Ước lượng năng lực hiện tại theta_t dựa trên lịch sử tương tác
            if step == 0:
                theta_t = 0.0  # Khởi tạo mặc định
                se_t = 0.5     # SE mặc định ban đầu
            else:
                # Xây dựng chuỗi tương tác cho model
                # Cần các tensor shape (1, max_seq_len, ...)
                x_emb_seq = torch.zeros(1, max_seq_len, 1024, device=device)
                x_feat_seq = torch.zeros(1, max_seq_len, 22, device=device)
                r_seq = torch.zeros(1, max_seq_len, device=device)
                T_time_seq = torch.zeros(1, max_seq_len, device=device)
                concept_indices_seq = torch.full((1, max_seq_len, max_c), -1, dtype=torch.long, device=device)
                padding_mask_seq = torch.zeros(1, max_seq_len, dtype=torch.bool, device=device)
                g_priors_seq = torch.zeros(1, max_seq_len, device=device)
                q_idx_seq = torch.zeros(1, max_seq_len, dtype=torch.long, device=device)
                
                # Điền lịch sử
                for t in range(step):
                    item_idx = selected_item_indices[t]
                    x_emb_seq[0, t] = q_embeddings_t[item_idx]
                    x_feat_seq[0, t] = q_features_t[item_idx]
                    r_seq[0, t] = r_history[t]
                    T_time_seq[0, t] = time_history[t]
                    
                    # Concepts
                    c_ids = q_concepts[item_idx]
                    concept_indices_seq[0, t, :len(c_ids)] = torch.tensor(c_ids, dtype=torch.long)
                    
                    padding_mask_seq[0, t] = True
                    g_priors_seq[0, t] = q_g_priors_t[item_idx]
                    q_idx_seq[0, t] = q_idxs_t[item_idx]
                    
                # Run model forward
                with torch.no_grad():
                    fused_x = model.model.fusion(x_emb_seq, x_feat_seq)
                    r_soft, g, s = model.model.refiner(fused_x, r_seq, g_priors_seq, q_indices=q_idx_seq)
                    I = model.model.embedding(fused_x, T_time_seq, r_soft)
                    h = model.model.sequence_model(I, padding_mask=padding_mask_seq)
                    theta_pred, log_var = model.model.decoder(h, concept_indices_seq, padding_mask=padding_mask_seq)
                    se = torch.exp(0.5 * log_var).squeeze(-1)
                    
                    # Tính targeted_ability tại bước step-1 (bước hiện tại cần chọn câu hỏi)
                    # Ta sẽ dự báo cho câu hỏi tiếp theo bằng cách lấy theta_pred tại vị trí t = step - 1
                    # Trong model thích ứng thực tế, năng lực học sinh được ước lượng tại bước cuối của lịch sử
                    last_step_idx = step - 1
                    
                    # Lấy embedding & feature của câu hỏi tiếp theo làm query giả định
                    # Ở đây ta lấy trung bình năng lực các concept của học sinh tại bước cuối
                    mastery = model.model.predictor.proj_mastery(theta_pred).squeeze(-1)
                    theta_t = mastery[0, last_step_idx].mean().item()
                    se_t = se[0, last_step_idx].item()
                    
            theta_history.append(theta_t)
            se_history.append(se_t)
            
            # 4.2. Chọn câu hỏi (Item Selection): Tìm câu hỏi có độ khó gần theta_t nhất và chưa làm
            candidate_indices = [i for i in range(pool_size) if i not in selected_item_indices]
            
            # Tính khoảng cách độ khó
            diffs = q_diffs[candidate_indices]
            best_cand_idx = np.argmin(np.abs(diffs - theta_t))
            chosen_item_idx = candidate_indices[best_cand_idx]
            
            selected_item_indices.append(chosen_item_idx)
            chosen_diff = q_diffs[chosen_item_idx]
            diff_history.append(chosen_diff)
            
            # 4.3. Giả lập kết quả làm bài r_t bằng mô hình 4PL lý thuyết
            g_val = q_gs[chosen_item_idx]
            s_val = q_ss[chosen_item_idx]
            
            # Xác suất đúng thực tế của học sinh
            P_correct = g_val + (1.0 - s_val - g_val) * (1.0 / (1.0 + np.exp(-(theta_true - chosen_diff))))
            
            # Lấy mẫu nhãn r_t
            r_t = 1.0 if np.random.rand() < P_correct else 0.0
            r_history.append(r_t)
            
            # Giả lập thời gian trả lời: nếu làm đúng thì thời gian ngẫu nhiên trung bình 15s, làm sai 25s
            T_time_val = np.random.exponential(15.0) + 5.0 if r_t == 1.0 else np.random.exponential(25.0) + 5.0
            T_time_val = np.clip(T_time_val, 2.0, 120.0)  # Lọc nhiễu
            time_history.append(T_time_val)
            
            # print(f"  Step {step+1:02d}: Chosen Diff={chosen_diff:.2f}, true_P={P_correct*100:.1f}%, r={int(r_t)} (Est Theta={theta_t:.2f})")

        results[name] = {
            'theta': np.array(theta_history),
            'diff': np.array(diff_history),
            'se': np.array(se_history),
            'r': np.array(r_history),
            'theta_true': theta_true,
            'color': student['color'],
            'marker': student['marker']
        }

    # =========================================================================
    # PLOT 1: Online Theta Convergence
    # =========================================================================
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        steps = np.arange(1, num_steps + 1)
        plt.plot(steps, res['theta'], label=f"Est: {name}", color=res['color'], marker=res['marker'], linewidth=2)
        plt.axhline(y=res['theta_true'], color=res['color'], linestyle='--', alpha=0.6, label=f"True: {name} ({res['theta_true']:.1f})")
        
    plt.title("Giả Lập Thích Ứng Trực Tuyến - Sự Hội Tụ Năng Lực Học Sinh (Online CAT Simulation)", fontsize=13, fontweight='bold')
    plt.xlabel("Số câu hỏi đã trả lời (Step t)", fontsize=11)
    plt.ylabel("Ước lượng năng lực (Theta)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-2.5, 2.5)
    
    # Xử lý legend trùng lặp
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plot1_path = os.path.join(artifacts_dir, "online_theta_convergence.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"Saved Online Plot 1 to {plot1_path}")

    # =========================================================================
    # PLOT 2: Online Adaptive Item Selection (3 Subplots)
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(11, 14), sharex=False)
    
    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        steps = np.arange(1, num_steps + 1)
        ax.plot(steps, res['theta'], label="Năng lực ước lượng (Est Theta)", color=res['color'], linewidth=2, zorder=1)
        ax.axhline(y=res['theta_true'], color='black', linestyle='--', alpha=0.7, label=f"Năng lực thực tế (True Theta: {res['theta_true']:.2f})", zorder=1)
        
        correct_mask = res['r'] == 1
        incorrect_mask = res['r'] == 0
        
        ax.scatter(steps[correct_mask], res['diff'][correct_mask], color='green', marker='o', s=80, label="Chọn câu Đúng (r=1)", zorder=2)
        ax.scatter(steps[incorrect_mask], res['diff'][incorrect_mask], color='red', marker='x', s=80, linewidths=2, label="Chọn câu Sai (r=0)", zorder=2)
        
        # Vẽ các nét dọc nối độ khó và năng lực
        for s_idx in range(num_steps):
            s = s_idx + 1
            ax.vlines(s, min(res['theta'][s_idx], res['diff'][s_idx]), max(res['theta'][s_idx], res['diff'][s_idx]), colors='gray', linestyles='dotted', alpha=0.4)
            
        ax.set_title(f"Quy trình Thích ứng Trực tuyến - {name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Giá trị năng lực / Độ khó", fontsize=10)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper left', fontsize=9)
        
    plt.xlabel("Số câu hỏi đã trả lời (Step t)", fontsize=11)
    plt.suptitle("Cơ Chế Tự Động Chọn Câu Hỏi Thích Ứng Trong Giả Lập Trực Tuyến", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot2_path = os.path.join(artifacts_dir, "online_adaptive_selection.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved Online Plot 2 to {plot2_path}")

    # =========================================================================
    # PLOT 3: Online SE Decay
    # =========================================================================
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        steps = np.arange(1, num_steps + 1)
        plt.plot(steps, res['se'], label=name, color=res['color'], marker=res['marker'], linewidth=2)
        
    plt.title("Sự Suy Giảm Độ Bất Định (SE Decay) Trong Giả Lập Trực Tuyến", fontsize=13, fontweight='bold')
    plt.xlabel("Số câu hỏi đã trả lời (Step t)", fontsize=11)
    plt.ylabel("Sai số chuẩn trích xuất (SE)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot3_path = os.path.join(artifacts_dir, "online_se_decay.png")
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    print(f"Saved Online Plot 3 to {plot3_path}")
    
    print("\nOnline CAT Simulation completed successfully!")

if __name__ == "__main__":
    main()
