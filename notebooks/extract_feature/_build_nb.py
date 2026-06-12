"""Builder: sinh feature_extraction.ipynb từ danh sách cell. Chạy 1 lần rồi xoá."""
import json, uuid, pathlib

cells = []
def md(src):  cells.append(("markdown", src))
def code(src): cells.append(("code", src))

# ---------------------------------------------------------------- Tiêu đề
md("""# Trích xuất đặc trưng cho dataset C++ (Content Database)

Notebook trích xuất đặc trưng theo tài liệu *Content_Database_v2_CodeOnly_Full.pdf* cho
dataset câu hỏi C++ dạng **output prediction / code tracing**.

- **Phase 2 — Surface features** (regex/AST, offline): #1–#14, #20–#21.
- **Phase 3 — LLM features** (gọi server LLM): #15–#23.
- Bỏ Phase 1 (embedding 768d). Output: `features.csv` + `features.json`.

Nguồn: `../prepare_dataset/questions_db_ready.json` (1393 câu).
Chạy bằng kernel **uv** của project. Thực hiện **từng cell một**.""")

# ---------------------------------------------------------------- B0 setup
md("## Bước 0 — Setup & load dữ liệu")
code('''import re
import json
import math
import statistics
import difflib
from pathlib import Path
from collections import Counter

import pandas as pd
import requests

# ----- Đường dẫn IO -----
HERE = Path.cwd()
DATA_PATH = (HERE / ".." / "prepare_dataset" / "questions_db_ready.json").resolve()
OUT_DIR = HERE.resolve()
print("DATA_PATH:", DATA_PATH, "| tồn tại:", DATA_PATH.exists())
print("OUT_DIR  :", OUT_DIR)

with open(DATA_PATH, encoding="utf-8") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
print("Số câu hỏi:", len(df))
df.head(3)''')

# ---------------------------------------------------------------- B1 tách code/text
md("""## Bước 1 — Tách code C++ khỏi câu hỏi tiếng Việt

`question_content` lẫn cả phần mô tả tiếng Việt và code. Heuristic: cắt tại `#include`
đầu tiên; nếu không có `#include` thì dò các dấu hiệu code (`int main`, `cout`, `{`...);
nếu không có code → `code=""` (câu lý thuyết).""")
code('''CODE_HINTS = ("int main", "void main", "cout", "cin", "using namespace",
              "printf", "scanf", "return 0", "#define")

def split_question(content: str):
    """Trả về (question_text, code)."""
    content = content or ""
    idx = content.find("#include")
    if idx != -1:
        return content[:idx].strip(), content[idx:].strip()
    # Không có #include: dò dấu hiệu code đầu tiên
    positions = [content.find(h) for h in CODE_HINTS if content.find(h) != -1]
    if positions:
        cut = min(positions)
        # lùi về đầu dòng chứa dấu hiệu để giữ trọn code
        nl = content.rfind("\\n", 0, cut)
        cut = nl + 1 if nl != -1 else cut
        return content[:cut].strip(), content[cut:].strip()
    return content.strip(), ""  # câu lý thuyết, không có code

df[["question_text", "code"]] = df["question_content"].apply(
    lambda c: pd.Series(split_question(c))
)

n_has_code = (df["code"].str.len() > 0).sum()
print(f"Câu có code: {n_has_code}/{len(df)}")
for qid in (7, 9, 10):
    row = df[df.question_id == qid].iloc[0]
    print(f"\\n===== Q{qid} =====\\n[text] {row.question_text!r}\\n[code]\\n{row.code}")''')

# ---------------------------------------------------------------- B2 lexical
md("""## Bước 2 — Surface lexical (mục 1A)

`L_qtok` (token câu hỏi), `L_lines` (số dòng code), `L_kw` (mật độ từ khóa C++),
`L_ids` (số identifier do người dùng định nghĩa).""")
code('''# Từ khóa C++ 3 tầng (theo PDF)
KW_T1 = {"int","float","double","char","bool","void","if","else","for","while",
         "do","return","cout","cin"}
KW_T2 = {"class","struct","enum","namespace","const","static"}
KW_T3 = {"virtual","override","template","friend","new","delete","dynamic_cast","typeid"}
CPP_KEYWORDS = KW_T1 | KW_T2 | KW_T3
# STL/identifier chuẩn cần loại khi đếm identifier người dùng
STL_NAMES = {"std","cout","cin","endl","string","vector","map","set","pair","main",
             "printf","scanf","include","iostream","using","namespace","system",
             "size_t","nullptr","NULL","true","false","this"}

WORD_RE = re.compile(r"[A-Za-z_]\\w*")

def code_tokens(code: str):
    return WORD_RE.findall(code)

def f_qtok(text: str) -> int:
    return len(text.split())

def f_lines(code: str) -> int:
    return len(code.strip().split("\\n")) if code.strip() else 0

def f_kw_density(code: str) -> float:
    toks = code_tokens(code)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in CPP_KEYWORDS) / len(toks)

def f_user_ids(code: str) -> int:
    toks = code_tokens(code)
    ids = {t for t in toks if t not in CPP_KEYWORDS and t not in STL_NAMES}
    return len(ids)

df["L_qtok"]  = df["question_text"].map(f_qtok)
df["L_lines"] = df["code"].map(f_lines)
df["L_kw"]    = df["code"].map(f_kw_density)
df["L_ids"]   = df["code"].map(f_user_ids)
df[["question_id","L_qtok","L_lines","L_kw","L_ids"]].head()''')

# ---------------------------------------------------------------- B3 syntactic
md("""## Bước 3 — Surface syntactic (mục 1B)

`S_nest` (độ sâu lồng ngoặc nhọn), `S_cf` (control-flow score),
`S_ops[6]` (vector tỷ lệ 6 nhóm toán tử đặc thù).""")
code('''def f_max_nesting(code: str) -> int:
    depth = max_depth = 0
    for ch in code:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth -= 1
    return max_depth

def _wcount(pattern, code):
    return len(re.findall(pattern, code))

def f_control_flow(code: str) -> float:
    n_if     = _wcount(r"\\b(if|else)\\b", code)
    n_for    = _wcount(r"\\bfor\\b", code)
    n_while  = _wcount(r"\\b(while|do)\\b", code)
    n_switch = _wcount(r"\\bswitch\\b", code)
    return n_if + 1.5 * n_for + 1.5 * n_while + 1.2 * n_switch

# 6 nhóm toán tử: [*, &, ->, ::, ++/--, =vs==]  (tỷ lệ trên tổng token code)
OP_PATTERNS = [
    ("d_ptr",    r"(?<![A-Za-z0-9_])\\*"),   # pointer / deref
    ("d_ref",    r"&(?!=)"),                  # reference / address
    ("d_arrow",  r"->"),                      # member ptr
    ("d_scope",  r"::"),                      # scope
    ("d_incr",   r"\\+\\+|--"),                # ++ / --
    ("d_assign", r"(?<![=!<>])=(?!=)"),       # gán = (không phải ==, !=, <=, >=)
]

def f_operator_vector(code: str):
    toks = code_tokens(code)
    denom = len(toks) if toks else 1
    return [len(re.findall(pat, code)) / denom for _name, pat in OP_PATTERNS]

df["S_nest"] = df["code"].map(f_max_nesting)
df["S_cf"]   = df["code"].map(f_control_flow)
df["S_ops"]  = df["code"].map(f_operator_vector)
df[["question_id","S_nest","S_cf","S_ops"]].head()''')

# ---------------------------------------------------------------- B4 structural
md("""## Bước 4 — Surface structural (mục 1C)

`T_class`, `T_oop[3]=[N_class,D_inherit,R_count]`,
`T_mem[4]=[is_stack,is_heap,is_static,is_global]`, `T_type` (mức 1–5).""")
code('''CLASS_RE = re.compile(r"\\b(class|struct)\\s+\\w+")

def f_class_count(code: str) -> int:
    return len(CLASS_RE.findall(code))

def f_oop_vector(code: str):
    n_class = f_class_count(code)
    # Độ sâu kế thừa: mỗi khai báo ": public/private/protected Base"
    inherit_edges = re.findall(r":\\s*(?:public|private|protected)\\s+\\w+", code)
    d_inherit = 1 if inherit_edges else 0   # heuristic: có kế thừa => sâu >=1
    # Quan hệ tổng: kế thừa + composition (class này chứa biến kiểu class khác) ~ xấp xỉ
    r_count = len(inherit_edges)
    return [n_class, d_inherit, r_count]

def f_memory_vector(code: str):
    is_heap   = 1 if re.search(r"\\b(new|malloc|calloc)\\b", code) else 0
    is_static = 1 if re.search(r"\\bstatic\\b", code) else 0
    # global: biến khai báo trước 'int main' (ngoài mọi hàm) — heuristic đơn giản
    head = code.split("int main")[0] if "int main" in code else ""
    is_global = 1 if re.search(r"^\\s*(int|float|double|char|bool)\\b", head, re.M) else 0
    is_stack  = 1 if code.strip() else 0    # có code => có biến stack
    return [is_stack, is_heap, is_static, is_global]

def f_type_complexity(code: str) -> int:
    if re.search(r"<\\s*\\w+.*?>|shared_ptr|unique_ptr|weak_ptr", code):
        return 5  # template / smart pointer
    if re.search(r"\\*\\s*\\*|\\w+\\s*\\*\\s*\\[|\\[\\s*\\]\\s*\\*", code):
        return 4  # pointer-to-pointer / array of pointer
    if re.search(r"[A-Za-z_]\\w*\\s*[*&]\\s*[A-Za-z_]|[*&]\\s*[A-Za-z_]\\w*\\s*=", code):
        return 3  # pointer / reference
    if re.search(r"\\[\\s*\\d*\\s*\\]|\\b(struct|enum)\\s+\\w+", code):
        return 2  # mảng / struct / enum
    if re.search(r"\\b(int|float|double|char|bool)\\b", code):
        return 1  # primitive
    return 0

df["T_class"] = df["code"].map(f_class_count)
df["T_oop"]   = df["code"].map(f_oop_vector)
df["T_mem"]   = df["code"].map(f_memory_vector)
df["T_type"]  = df["code"].map(f_type_complexity)
df[["question_id","T_class","T_oop","T_mem","T_type"]].head()''')

# ---------------------------------------------------------------- B5 option
md("""## Bước 5 — Option features (mục 1D)

`O_var` (variance độ dài option), `O_spc[3]=[CompileError,RuntimeError,AnotherAnswer]`,
`O_sim` (độ tương đồng giữa các option thuần số).""")
code('''def f_option_len_var(options):
    lens = [len(str(o)) for o in options]
    return statistics.pvariance(lens) if len(lens) > 1 else 0.0

def f_special_flags(options):
    low = [str(o).strip().lower() for o in options]
    return [
        int(any("compile error" in o for o in low)),
        int(any("runtime error" in o for o in low)),
        int(any("another answer" in o for o in low)),
    ]

def _is_numeric(o):
    return str(o).strip().lstrip("-").replace(" ", "").isdigit()

def f_numeric_similarity(options):
    nums = [str(o).strip() for o in options if _is_numeric(o)]
    if len(nums) < 2:
        return 0.0
    sims = [
        difflib.SequenceMatcher(None, a, b).ratio()
        for i, a in enumerate(nums) for b in nums[i + 1:]
    ]
    return sum(sims) / len(sims)

df["O_var"] = df["all_options_content"].map(f_option_len_var)
df["O_spc"] = df["all_options_content"].map(f_special_flags)
df["O_sim"] = df["all_options_content"].map(f_numeric_similarity)
df[["question_id","O_var","O_spc","O_sim"]].head()''')

# ---------------------------------------------------------------- B6 sanity + checkpoint
md("""## Bước 6 — Sanity-check & lưu surface checkpoint

Đối chiếu câu #7 và #10 với ví dụ PHẦN 4 của PDF, rồi lưu `features_surface.{csv,json}`
để Phase 3 có thể chạy độc lập.""")
code('''check_cols = ["question_id","L_lines","S_nest","S_cf","S_ops","T_type","O_sim"]
print("Đối chiếu PDF (PHẦN 4):")
print(df[df.question_id.isin([7, 10])][check_cols].to_string(index=False))
# Kỳ vọng: Q7 L_lines nhỏ, S_nest=0 ; Q10 S_nest=1, S_cf=1.5

def explode_vectors(frame):
    """Tách các cột vector thành nhiều cột số cho CSV."""
    out = frame.copy()
    vec_cols = {
        "S_ops": ["d_ptr","d_ref","d_arrow","d_scope","d_incr","d_assign"],
        "T_oop": ["oop_nclass","oop_dinherit","oop_rcount"],
        "T_mem": ["mem_stack","mem_heap","mem_static","mem_global"],
        "O_spc": ["spc_compile","spc_runtime","spc_another"],
    }
    for col, names in vec_cols.items():
        if col in out.columns:
            mat = pd.DataFrame(out[col].tolist(), index=out.index, columns=names)
            out = pd.concat([out.drop(columns=[col]), mat], axis=1)
    return out

SURFACE_COLS = ["question_id","L_qtok","L_lines","L_kw","L_ids","S_nest","S_cf",
                "S_ops","T_class","T_oop","T_mem","T_type","O_var","O_spc","O_sim"]
surface = df[SURFACE_COLS].copy()
surface.to_json(OUT_DIR / "features_surface.json", orient="records",
                force_ascii=False, indent=2)
explode_vectors(surface).to_csv(OUT_DIR / "features_surface.csv", index=False)
print("\\nĐã lưu features_surface.csv / .json")''')

# ---------------------------------------------------------------- B7 LLM client
md("""## Bước 7 — LLM client + prompt hợp nhất (Phase 3)

Gọi thẳng server OpenAI-compatible bằng `requests`. Model là **reasoning model**
(trả `reasoning_content` riêng, đáp án nằm ở `content`) nên dùng `max_tokens` lớn và
parse JSON từ `content` (fallback regex). Một prompt hợp nhất/câu trả về đủ trường LLM
để giảm số lần gọi.""")
code('''LLM_BASE = "https://llm.phuocnguyn.id.vn/v1"
LLM_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"
LLM_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-dummy",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

_JSON_RE = re.compile(r"\\{.*\\}", re.S)

def _parse_json(text: str):
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):                       # bỏ code-fence ```json ... ```
        t = re.sub(r"^```[a-zA-Z]*\\n?|\\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        m = _JSON_RE.search(t)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

# Lưu ý: đây là reasoning model — tốn nhiều token cho phần suy luận (reasoning_content),
# đáp án JSON nằm ở 'content'. Cần max_tokens lớn, nếu không content sẽ rỗng (finish=length).
def call_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 6144,
             retries: int = 2):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system",
             "content": "Bạn là chuyên gia phân tích độ khó câu hỏi lập trình C++."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(f"{LLM_BASE}/chat/completions", headers=LLM_HEADERS,
                              json=payload, timeout=120)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            parsed = _parse_json(content)
            if parsed is not None:
                return parsed
            last_err = "parse-fail"
        except Exception as e:
            last_err = str(e)
    return {"_error": last_err}

PROMPT_TEMPLATE = """Cho đoạn code C++ và các phương án trả lời sau.

CODE:
{code}

ĐÁP ÁN ĐÚNG: {correct}
CÁC PHƯƠNG ÁN: {options}

Hãy phân tích trong đầu (trace từng bước thực thi), sau đó CHỈ trả về một JSON duy nhất theo schema:
{{
  "step_count": <int, số bước trace nguyên tử>,
  "reasoning_depth": <int, độ sâu suy luận>,
  "weighted_steps": <number, tổng bước có trọng số: Identity*1.0, Retrieval*1.5, Simulation*1.0, Constraint*1.2>,
  "ambiguity": "<low|medium|high>",
  "bloom_level": <int 1-6: 1 Remember,2 Understand,3 Apply,4 Analyze,5 Evaluate,6 Create>,
  "misconceptions": [{{"type":"<Conceptual|Procedural|Side-effect trap|Syntactic trap|Lack of sense>","weight":<number>}}],
  "distractor_scores": [{{"option":"<nội dung phương án sai>","score":<int 1-5>,"trap_type":"<mô tả lỗi trace>"}}]
}}
Chỉ xuất JSON, không thêm chữ nào khác."""

def build_prompt(row):
    correct_ids = set(row["correct_option_ids"])
    ids = row["all_option_ids"]
    opts = row["all_options_content"]
    correct = ", ".join(str(o) for i, o in zip(ids, opts) if i in correct_ids)
    return PROMPT_TEMPLATE.format(
        code=row["code"] or row["question_content"],
        correct=correct,
        options=" | ".join(str(o) for o in opts),
    )

# Test thử 1 câu (Q10) để xác nhận parse được
_demo = call_llm(build_prompt(df[df.question_id == 10].iloc[0]), temperature=0.0)
print(json.dumps(_demo, ensure_ascii=False, indent=2))''')

# ---------------------------------------------------------------- B8 run with cache
md("""## Bước 8 — Chạy Phase 3 có cache/resume

Mỗi câu gọi 3 temperature (0.0, 0.3, 0.7). Kết quả mỗi (question_id, temperature) ghi vào
`llm_cache.jsonl` để chạy lại không gọi trùng. Đặt `LIMIT` để chạy thử trước
(VD: `LIMIT=5`), rồi đặt `LIMIT=None` để chạy toàn bộ.""")
code('''from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_PATH = OUT_DIR / "llm_cache.jsonl"
TEMPERATURES = [0.0, 0.3, 0.7]
LIMIT = 5            # <<< đổi thành None để chạy toàn bộ 1393 câu
MAX_WORKERS = 4

def load_cache():
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[(rec["question_id"], rec["temperature"])] = rec["result"]
    return cache

cache = load_cache()
print("Đã có trong cache:", len(cache), "bản ghi")

work_df = df if LIMIT is None else df.head(LIMIT)
tasks = [(int(r.question_id), t, r) for _, r in work_df.iterrows()
         for t in TEMPERATURES if (int(r.question_id), t) not in cache]
print("Cần gọi LLM:", len(tasks), "lần")

def _run(qid, temp, row):
    res = call_llm(build_prompt(row), temperature=temp)
    return qid, temp, res

cache_file = open(CACHE_PATH, "a", encoding="utf-8")
try:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_run, qid, t, r) for qid, t, r in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            qid, temp, res = fut.result()
            cache[(qid, temp)] = res
            if isinstance(res, dict) and "_error" not in res:
                # chỉ ghi xuống file những kết quả thành công -> lần chạy sau
                # tự động thử lại các câu lỗi (không nằm trong cache đã load)
                cache_file.write(json.dumps(
                    {"question_id": qid, "temperature": temp, "result": res},
                    ensure_ascii=False) + "\\n")
                cache_file.flush()
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} xong")
finally:
    cache_file.close()
print("Hoàn tất. Tổng cache:", len(cache))''')

# ---------------------------------------------------------------- B9 aggregate LLM
md("""## Bước 9 — Tổng hợp feature LLM từ cache

Với mỗi câu: số → mean qua 3 temperature; category (ambiguity, bloom) → mode; độ chênh
giữa 3 lần đóng góp vào `H_amb`. Suy ra
`H_N,H_D,H_W,H_amb,H_B,H_M,H_P,H_Dmax,H_Dmean`.""")
code('''MISC_WEIGHTS = {
    "conceptual": 0.8, "procedural": 0.5, "side-effect trap": 1.0,
    "syntactic trap": 0.6, "lack of sense": 1.2,
}
AMB_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}

def _num(d, key, default=0.0):
    try:
        v = float(d.get(key))
        return v if math.isfinite(v) else default
    except Exception:
        return default

def aggregate_llm(qid):
    runs = [cache.get((qid, t)) for t in TEMPERATURES]
    runs = [r for r in runs if isinstance(r, dict) and "_error" not in r]
    if not runs:
        return None
    H_N = statistics.mean(_num(r, "step_count") for r in runs)
    H_D = statistics.mean(_num(r, "reasoning_depth") for r in runs)
    W_list = [_num(r, "weighted_steps") for r in runs]
    H_W = statistics.mean(W_list)
    # Bloom: mode
    blooms = [int(_num(r, "bloom_level")) for r in runs if _num(r, "bloom_level") > 0]
    H_B = statistics.mode(blooms) if blooms else 0
    # Ambiguity: trung bình map + cộng độ phân tán của W (chênh lệch lớn => mơ hồ)
    amb_self = statistics.mean(
        AMB_MAP.get(str(r.get("ambiguity", "")).lower(), 0.0) for r in runs)
    w_spread = (statistics.pstdev(W_list) / (statistics.mean(W_list) + 1e-9)
                if len(W_list) > 1 else 0.0)
    H_amb = round(min(1.0, amb_self * 0.7 + min(w_spread, 1.0) * 0.3), 4)
    # Misconceptions: lấy run đầu có danh sách
    miscs = next((r.get("misconceptions") for r in runs
                  if isinstance(r.get("misconceptions"), list) and r["misconceptions"]), [])
    H_M = len(miscs)
    H_P = sum(MISC_WEIGHTS.get(str(m.get("type","")).strip().lower(),
                               _num(m, "weight")) for m in miscs)
    # Distractor plausibility
    scores = []
    for r in runs:
        for d in (r.get("distractor_scores") or []):
            s = _num(d, "score")
            if s > 0:
                scores.append(s)
    H_Dmax = max(scores) if scores else 0.0
    H_Dmean = statistics.mean(scores) if scores else 0.0
    return dict(question_id=qid, H_N=round(H_N,3), H_D=round(H_D,3), H_W=round(H_W,3),
                H_amb=H_amb, H_B=H_B, H_M=H_M, H_P=round(H_P,3),
                H_Dmax=H_Dmax, H_Dmean=round(H_Dmean,3))

llm_rows = [a for qid in df.question_id if (a := aggregate_llm(int(qid))) is not None]
llm_df = pd.DataFrame(llm_rows)
print("Số câu có feature LLM:", len(llm_df))
llm_df.head()''')

# ---------------------------------------------------------------- B10 merge + export
md("""## Bước 10 — Gộp toàn bộ & xuất `features.csv` + `features.json`""")
code('''full = df[SURFACE_COLS].merge(llm_df, on="question_id", how="left")

# JSON: giữ cấu trúc lồng (S_ops, T_oop... là list)
full.to_json(OUT_DIR / "features.json", orient="records",
             force_ascii=False, indent=2)
# CSV: phẳng hoá các vector con
explode_vectors(full).to_csv(OUT_DIR / "features.csv", index=False)
print("Đã lưu features.csv / features.json:", full.shape)

# Sanity-check cuối: Q10 phải khó hơn Q7 (H_W lớn hơn) — theo nhận định PDF
cmp_cols = ["question_id","L_lines","S_nest","S_cf","H_W","H_B","H_Dmean"]
print(full[full.question_id.isin([7,10])][cmp_cols].to_string(index=False))
full.describe(include="all").T''')

# ---------------------------------------------------------------- ghi file
nb = {
    "cells": [
        {
            "cell_type": t,
            "metadata": {},
            "source": s.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if t == "code" else {}),
            "id": uuid.uuid4().hex[:8],
        }
        for t, s in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = pathlib.Path(__file__).parent / "feature_extraction.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", out, "with", len(cells), "cells")
