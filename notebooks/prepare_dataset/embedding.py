import numpy as np
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
import re

# ===== THIẾT LẬP MÔI TRƯỜNG =====
EMB_MODEL = "microsoft/graphcodebert-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_BATCH = 8 if DEVICE == "cpu" else 32     # CPU dùng batch nhỏ, GPU tăng lên 32/64 tùy VRAM
EMB_MAXLEN = 512                              # CodeBERT max context
OUT_DIR = Path("./output_embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)    # Tự động tạo thư mục nếu chưa có

print(f"🚀 Khởi động trích xuất đặc trưng CodeBERT trên thiết bị: {DEVICE.upper()}")

# ===== TẢI MÔ HÌNH VÀ TOKENIZER =====
_tok = AutoTokenizer.from_pretrained(EMB_MODEL)

# Lách an toàn: codebert-base chỉ có pytorch_model.bin (chặn CVE-2025-32434 trên torch < 2.6)
try:
    _emb_model = AutoModel.from_pretrained(EMB_MODEL)
except Exception as e:
    print(f"⚠️ from_pretrained bị chặn ({type(e).__name__}), chuyển sang load_state_dict thủ công...")
    _emb_model = AutoModel.from_config(AutoConfig.from_pretrained(EMB_MODEL))
    _sd = torch.load(hf_hub_download(EMB_MODEL, "pytorch_model.bin"),
                     map_location="cpu", weights_only=True)
    _miss, _unexp = _emb_model.load_state_dict(_sd, strict=False)
    assert not _unexp, f"unexpected keys: {list(_unexp)[:5]}"

_emb_model = _emb_model.to(DEVICE).eval()
print("✅ Tải CodeBERT thành công!")

# ===== HÀM TIỀN XỬ LÝ NGỮ CẢNH (FULL-CONTEXT) =====
def safe_clean_cpp(code_text):
    """
    Làm sạch mã nguồn C++: Loại bỏ comment và chuẩn hóa khoảng trắng.
    Tuyệt đối giữ lại #include, namespace và main() để không làm hỏng logic bẫy lỗi.
    """
    if not isinstance(code_text, str) or not code_text.strip():
        return ""

    # 1. Xóa comment nhiều dòng: /* ... */
    # (Dùng DOTALL để khớp qua cả các ký tự xuống dòng)
    cleaned = re.sub(r'/\*[\s\S]*?\*/', '', code_text)
    
    # 2. Xóa comment một dòng: // ...
    cleaned = re.sub(r'//.*', '', cleaned)
    
    # 3. Nén nhiều dòng trống liên tiếp thành 1 dòng (Tối ưu token)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    # 4. Cắt khoảng trắng dư thừa ở hai đầu
    return cleaned.strip()

def build_full_context(row):
    """
    Gộp đề bài, mã nguồn (đã được làm sạch) và các đáp án.
    """
    # Lấy nội dung câu hỏi và làm sạch ngay lập tức
    raw_content = str(row.get("question_content", ""))
    clean_content = safe_clean_cpp(raw_content)
    
    # Xử lý danh sách đáp án
    options_list = row.get("all_options_content", [])
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    formatted_options = []
    if isinstance(options_list, list):
        for i, opt_text in enumerate(options_list):
            label = labels[i] if i < len(labels) else str(i)
            formatted_options.append(f"[{label}] {opt_text}")
            
    options_text = " | ".join(formatted_options)
    
    # Nối lại với các từ khóa hướng dẫn Attention
    if options_text:
        return f"Question: {clean_content}\nOptions: {options_text}"
    return f"Question: {clean_content}"

# ===== HÀM NHÚNG DỮ LIỆU (EMBEDDING) =====
@torch.no_grad()
def embed_codes(texts, batch_size=EMB_BATCH):
    vecs = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        # Lọc text rỗng (nếu có) thành khoảng trắng để tránh lỗi Tokenizer
        chunk = [t if t.strip() else " " for t in texts[i:i + batch_size]]
        
        # Tokenize
        enc = _tok(chunk, padding=True, truncation=True, max_length=EMB_MAXLEN, return_tensors="pt").to(DEVICE)
        
        # Chạy qua model
        out = _emb_model(**enc).last_hidden_state            # Shape: [B, L, 768]
        mask = enc["attention_mask"].unsqueeze(-1).float()   # Shape: [B, L, 1]
        
        # Mean Pooling theo attention mask (bỏ qua các token padding [PAD])
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vecs.append(emb.cpu().numpy().astype("float32"))
        
        # In tiến trình
        done = min(i + batch_size, total)
        if (i // batch_size) % 10 == 0 or done == total:
            print(f"⏳ Đang xử lý: {done}/{total} câu hỏi...")
            
    return np.vstack(vecs)

# ===== THỰC THI =====
if __name__ == "__main__":
    # 1. Đọc dữ liệu (Thay đường dẫn bằng file JSON thực tế của bạn)
    # Giả sử file chứa mảng JSON như bạn đã cung cấp
    print("\n📂 Đang tải dữ liệu JSON...")
    df = pd.read_json(r"D:\IntelligentTesting\intelligent-testing\notebooks\prepare_dataset\questions_db_ready.json") 
    
    # 2. Xây dựng Full-Context cho toàn bộ DataFrame
    print("🧠 Đang xây dựng Full-Context (Đề bài + Code + Đáp án)...")
    emb_input = [build_full_context(row) for _, row in df.iterrows()]
    
    # 3. Chạy Embedding
    print("⚙️ Bắt đầu trích xuất Vector (Embedding)...")
    E_code = embed_codes(emb_input)
    
    # 4. Kiểm tra và Lưu trữ
    print(f"\n📊 Kích thước ma trận Code Embedding: {E_code.shape}") # Kỳ vọng: (N, 768)
    
    np.save(OUT_DIR / "graph_code_embeddings.npy", E_code)
    np.save(OUT_DIR / "graph_code_embeddings_qid.npy", df["question_id"].to_numpy())
    
    # Đưa ngược lại vào DataFrame để chuẩn bị cho bước gộp Tabular Data (nếu cần)
    df["E_code"] = list(E_code)
    
    print(f"🎉 Hoàn tất! Đã lưu file tại {OUT_DIR}/")