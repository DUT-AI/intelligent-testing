import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question, StudentSession
from scripts.train_neural_cat import StudentSequenceDataset

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
    
    # 2. Load Test Dataset from Database
    print("Loading test data from database...")
    db_session = SessionLocal()
    try:
        from sqlalchemy.orm import joinedload
        db_questions = (
            db_session.query(Question)
            .options(joinedload(Question.features), joinedload(Question.misconceptions))
            .all()
        )
        
        question_embeddings = {}
        question_concepts = {}
        question_features = {}
        question_option_counts = {}
        
        for q in db_questions:
            if q.embedding is not None:
                question_embeddings[q.id] = np.array(q.embedding, dtype=np.float32)
            question_concepts[q.id] = q.concept_ids or []
            question_option_counts[q.id] = q.option_count or 0
            
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
                
            question_features[q.id] = np.array(feat_vec, dtype=np.float32)
            
        test_sequences = (
            db_session.query(StudentSession)
            .filter(StudentSession.dataset_type == "test")
            .all()
        )
    finally:
        db_session.close()
        
    hparams = model.hparams
    max_seq_len = int(hparams["max_seq_len"] if isinstance(hparams, dict) else getattr(hparams, "max_seq_len"))
    
    test_dataset = StudentSequenceDataset(
        sequences=test_sequences,
        question_embeddings=question_embeddings,
        question_concepts=question_concepts,
        question_features=question_features,
        question_option_counts=question_option_counts,
        max_seq_len=max_seq_len
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 3. Find 3 representative student sessions
    print("Finding representative students...")
    student_candidates = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            x, x_feat, r, T_time, concept_indices, padding_mask, g_priors = batch
            
            # Tính độ dài chuỗi thực tế
            seq_len = padding_mask.sum().item()
            if seq_len < 25 or seq_len > 50:
                continue
                
            # Đưa lên device
            x = x.to(device)
            x_feat = x_feat.to(device)
            r = r.to(device)
            T_time = T_time.to(device)
            concept_indices = concept_indices.to(device)
            padding_mask = padding_mask.to(device)
            g_priors = g_priors.to(device)
            
            logits, g, s, se = model(x, x_feat, r, T_time, concept_indices, padding_mask, g_priors)
            P = torch.sigmoid(logits)
            preds = (P >= 0.50).float()
            
            correct = (preds[padding_mask] == r[padding_mask]).float()
            acc = correct.mean().item()
            
            student_candidates.append({
                'idx': idx,
                'acc': acc,
                'seq_len': seq_len,
                'batch': batch
            })
            
            if len(student_candidates) > 500:  # Đủ mẫu để chọn
                break
                
    # Chọn học sinh giỏi (accuracy cao)
    giol_candidates = [s for s in student_candidates if s['acc'] >= 0.85]
    giol = giol_candidates[0] if giol_candidates else sorted(student_candidates, key=lambda s: -s['acc'])[0]
    
    # Chọn học sinh yếu (accuracy thấp)
    yeu_candidates = [s for s in student_candidates if s['acc'] <= 0.45]
    yeu = yeu_candidates[0] if yeu_candidates else sorted(student_candidates, key=lambda s: s['acc'])[0]
    
    # Chọn học sinh trung bình (accuracy ~ 0.65)
    tb_candidates = [s for s in student_candidates if 0.60 <= s['acc'] <= 0.70]
    tb = tb_candidates[0] if tb_candidates else sorted(student_candidates, key=lambda s: abs(s['acc'] - 0.65))[0]
    
    print(f"Selected Students:")
    print(f"  - Giỏi: Acc={giol['acc']:.2f}, SeqLen={giol['seq_len']}")
    print(f"  - Yếu: Acc={yeu['acc']:.2f}, SeqLen={yeu['seq_len']}")
    print(f"  - Trung bình: Acc={tb['acc']:.2f}, SeqLen={tb['seq_len']}")
    
    # 4. Extract detailed histories for the three students
    def extract_student_data(student_info):
        batch = student_info['batch']
        x, x_feat, r, T_time, concept_indices, padding_mask, g_priors = batch
        seq_len = student_info['seq_len']
        
        x = x.to(device)
        x_feat = x_feat.to(device)
        r = r.to(device)
        T_time = T_time.to(device)
        concept_indices = concept_indices.to(device)
        padding_mask = padding_mask.to(device)
        g_priors = g_priors.to(device)
        
        with torch.no_grad():
            # Chạy qua model
            # 1. Feature Fusion
            fused_x = model.model.fusion(x, x_feat)
            
            # 2. Refiner
            r_soft, g, s = model.model.refiner(fused_x, r, g_priors)
            
            # 3. Embeddings
            I = model.model.embedding(fused_x, T_time, r_soft)
            
            # 4. Sequence Model
            h = model.model.sequence_model(I, padding_mask=padding_mask)
            
            # 5. Decoder
            theta_pred, log_var = model.model.decoder(h, concept_indices, padding_mask=padding_mask)
            se = torch.exp(0.5 * log_var).squeeze(-1)
            
            # 6. Predictor
            # Trích xuất targeted_ability & difficulty thủ công để theo dõi
            query = model.model.predictor.proj_q(fused_x)
            key = model.model.predictor.proj_k(theta_pred)
            scores = torch.matmul(key, query.unsqueeze(-1)).squeeze(-1) / (model.model.predictor.d_h**0.5)
            
            valid_mask = concept_indices != -1
            scores = scores.masked_fill(~valid_mask, -1e9)
            beta = torch.softmax(scores, dim=-1)
            
            mastery = model.model.predictor.proj_mastery(theta_pred).squeeze(-1)
            targeted_ability = (beta * mastery).sum(dim=-1)
            
            difficulty = model.model.predictor.proj_diff(fused_x).squeeze(-1)
            
            # Đưa về numpy và lọc theo độ dài thực tế
            theta_seq = targeted_ability[0, :seq_len].cpu().numpy()
            diff_seq = difficulty[0, :seq_len].cpu().numpy()
            se_seq = se[0, :seq_len].cpu().numpy()
            r_seq = r[0, :seq_len].cpu().numpy()
            concepts_seq = concept_indices[0, :seq_len].cpu().numpy()
            mastery_seq = mastery[0, :seq_len].cpu().numpy()
            
            return {
                'theta': theta_seq,
                'diff': diff_seq,
                'se': se_seq,
                'r': r_seq,
                'concepts': concepts_seq,
                'mastery': mastery_seq
            }
            
    giol_data = extract_student_data(giol)
    tb_data = extract_student_data(tb)
    yeu_data = extract_student_data(yeu)
    
    # =========================================================================
    # PLOT 1: Ability State Theta Convergence
    # =========================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(giol_data['theta'], label=f"Học sinh Giỏi (Acc: {giol['acc']:.2f})", color='green', marker='o', linewidth=2)
    plt.plot(tb_data['theta'], label=f"Học sinh Trung bình (Acc: {tb['acc']:.2f})", color='blue', marker='s', linewidth=2)
    plt.plot(yeu_data['theta'], label=f"Học sinh Yếu (Acc: {yeu['acc']:.2f})", color='red', marker='^', linewidth=2)
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
    plt.title("Sự Hội Tụ Năng Lực Nổi Bật Của Học Sinh Qua Từng Bước CAT", fontsize=14, fontweight='bold')
    plt.xlabel("Số câu hỏi đã trả lời (Step t)", fontsize=12)
    plt.ylabel("Ước lượng năng lực (Theta)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot1_path = os.path.join(artifacts_dir, "theta_convergence.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"Saved Plot 1 to {plot1_path}")
    
    # =========================================================================
    # PLOT 2: Adaptive Item Selection (3 Subplots)
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(11, 14), sharex=False)
    
    def plot_adaptive(ax, data, title):
        steps = np.arange(len(data['theta']))
        ax.plot(steps, data['theta'], label="Năng lực (Theta)", color='blue', linewidth=2, zorder=1)
        
        # Tách các điểm Đúng (r=1) và Sai (r=0)
        correct_mask = data['r'] == 1
        incorrect_mask = data['r'] == 0
        
        ax.scatter(steps[correct_mask], data['diff'][correct_mask], color='green', marker='o', s=80, label="Chọn câu Đúng (r=1)", zorder=2)
        ax.scatter(steps[incorrect_mask], data['diff'][incorrect_mask], color='red', marker='x', s=80, linewidths=2, label="Chọn câu Sai (r=0)", zorder=2)
        
        # Vẽ các nét nối từ độ khó đến năng lực để thể hiện thích ứng
        for s in steps:
            ax.vlines(s, min(data['theta'][s], data['diff'][s]), max(data['theta'][s], data['diff'][s]), colors='gray', linestyles='dotted', alpha=0.4)
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel("Giá trị", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper left', fontsize=9)
        
    plot_adaptive(axes[0], giol_data, f"Mô phỏng Thích ứng - Học sinh Giỏi (Độ chính xác: {giol['acc']*100:.1f}%)")
    plot_adaptive(axes[1], tb_data, f"Mô phỏng Thích ứng - Học sinh Trung bình (Độ chính xác: {tb['acc']*100:.1f}%)")
    plot_adaptive(axes[2], yeu_data, f"Mô phỏng Thích ứng - Học sinh Yếu (Độ chính xác: {yeu['acc']*100:.1f}%)")
    
    plt.xlabel("Số câu hỏi đã trả lời (Step t)", fontsize=11)
    plt.suptitle("Cơ Chế Lựa Chọn Câu Hỏi Thích Ứng (Adaptive Item Selection)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot2_path = os.path.join(artifacts_dir, "adaptive_selection.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved Plot 2 to {plot2_path}")
    
    # =========================================================================
    # PLOT 3: Uncertainty SE Decay
    # =========================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(giol_data['se'], label=f"Học sinh Giỏi (Acc: {giol['acc']:.2f})", color='green', marker='o', linewidth=2)
    plt.plot(tb_data['se'], label=f"Học sinh Trung bình (Acc: {tb['acc']:.2f})", color='blue', marker='s', linewidth=2)
    plt.plot(yeu_data['se'], label=f"Học sinh Yếu (Acc: {yeu['acc']:.2f})", color='red', marker='^', linewidth=2)
    plt.title("Sự Suy Giảm Độ Bất Định (Standard Error SE Decay) Theo Thời Gian", fontsize=14, fontweight='bold')
    plt.xlabel("Số câu hỏi đã trả lời (Step t)", fontsize=12)
    plt.ylabel("Sai số chuẩn trích xuất (SE)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot3_path = os.path.join(artifacts_dir, "se_decay.png")
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    print(f"Saved Plot 3 to {plot3_path}")
    
    # =========================================================================
    # PLOT 4: Concept Mastery Profile (Bar Plot comparison)
    # =========================================================================
    # Lấy 5 concept xuất hiện phổ biến nhất ở cuối chuỗi của học sinh Trung bình
    last_concepts = tb_data['concepts'][-1]
    valid_c = [c for c in last_concepts if c != -1][:5]
    if len(valid_c) < 5:
        # Fallback lấy các concept cố định
        valid_c = [1, 2, 3, 4, 5]
        
    concept_names = [f"Concept {c}" for c in valid_c]
    
    # Lấy mastery cuối cùng của 3 học sinh cho các concept này
    def get_last_mastery(data, concept_list, valid_indices):
        last_step = -1
        # Tìm chỉ số tương ứng trong list concept của bước cuối
        masteries = []
        for c in valid_indices:
            idx_in_step = np.where(concept_list[last_step] == c)[0]
            if len(idx_in_step) > 0:
                masteries.append(data['mastery'][last_step, idx_in_step[0]])
            else:
                # Nếu không tìm thấy, lấy trung bình các concept có sẵn của học sinh đó
                masteries.append(np.mean(data['mastery'][last_step]))
        return masteries
        
    giol_m = get_last_mastery(giol_data, giol_data['concepts'], valid_c)
    tb_m = get_last_mastery(tb_data, tb_data['concepts'], valid_c)
    yeu_m = get_last_mastery(yeu_data, yeu_data['concepts'], valid_c)
    
    # Vẽ Bar Chart nhóm
    x_indices = np.arange(len(valid_c))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x_indices - width, giol_m, width, label="Học sinh Giỏi", color='green', alpha=0.8)
    plt.bar(x_indices, tb_m, width, label="Học sinh Trung bình", color='blue', alpha=0.8)
    plt.bar(x_indices + width, yeu_m, width, label="Học sinh Yếu", color='red', alpha=0.8)
    
    plt.title("Chẩn Đoán Hồ Sơ Kiến Thức Chi Tiết (Concept Mastery Profile)", fontsize=14, fontweight='bold')
    plt.xlabel("Mảng kiến thức / Khái niệm (Concepts)", fontsize=12)
    plt.ylabel("Mức độ làm chủ (Mastery Score)", fontsize=12)
    plt.xticks(x_indices, concept_names, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5, axis='y')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot4_path = os.path.join(artifacts_dir, "concept_mastery.png")
    plt.savefig(plot4_path, dpi=150)
    plt.close()
    print(f"Saved Plot 4 to {plot4_path}")
    
    # =========================================================================
    # PLOT 5: Item Characteristic Curves (ICC) 4PL IRT
    # =========================================================================
    theta_range = np.linspace(-3.0, 3.0, 200)
    
    # Định nghĩa 3 câu hỏi tiêu biểu: Dễ, Trung bình, Khó
    # Giả định tham số g và s thực tế (ví dụ g=0.25 cho trắc nghiệm, s=0.05)
    g = 0.25
    s = 0.05
    
    def irt_4pl(theta, d, g, s):
        # P(theta) = g + (1 - s - g) * sigmoid(theta - d)
        return g + (1.0 - s - g) * (1.0 / (1.0 + np.exp(-(theta - d))))
        
    plt.figure(figsize=(10, 6))
    plt.plot(theta_range, irt_4pl(theta_range, -1.5, g, s), label="Câu hỏi Dễ (Độ khó d = -1.5)", color='green', linewidth=2.5)
    plt.plot(theta_range, irt_4pl(theta_range, 0.0, g, s), label="Câu hỏi Trung bình (Độ khó d = 0.0)", color='blue', linewidth=2.5)
    plt.plot(theta_range, irt_4pl(theta_range, 1.5, g, s), label="Câu hỏi Khó (Độ khó d = 1.5)", color='red', linewidth=2.5)
    
    # Vẽ tiệm cận
    plt.axhline(y=g, color='black', linestyle=':', alpha=0.5)
    plt.text(-2.9, g + 0.02, f"Tiệm cận đoán mò g = {g:.2f}", fontsize=9, color='black', alpha=0.7)
    plt.axhline(y=1.0 - s, color='black', linestyle=':', alpha=0.5)
    plt.text(-2.9, 1.0 - s - 0.04, f"Tiệm cận trượt chân s = {s:.2f} (1-s = {1.0-s:.2f})", fontsize=9, color='black', alpha=0.7)
    
    plt.title("Đường Cong Đặc Trưng Câu Hỏi 4PL (Item Characteristic Curves - ICC)", fontsize=14, fontweight='bold')
    plt.xlabel("Năng lực học sinh (Theta)", fontsize=12)
    plt.ylabel("Xác suất trả lời đúng - P(Theta)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plot5_path = os.path.join(artifacts_dir, "icc_curves.png")
    plt.savefig(plot5_path, dpi=150)
    plt.close()
    print(f"Saved Plot 5 to {plot5_path}")
    
    print("\nVisualizations successfully generated and saved to artifacts directory!")

if __name__ == "__main__":
    main()
