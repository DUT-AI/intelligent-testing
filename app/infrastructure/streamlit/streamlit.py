# pylint: disable=no-member
import glob
import os
import sys
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.infrastructure.database.cat_repository import SQLAlchemyQuestionRepository
from app.infrastructure.models.neural_cat_adapter import NeuralCATModelAdapter
from app.use_cases.cat.start_session import StartCATSessionUseCase
from app.use_cases.cat.submit_answer import SubmitAnswerUseCase

# Set Streamlit Page Config
st.set_page_config(
    page_title="Hệ thống Khảo thí Thích ứng (NeuralCAT)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Mode, Glassmorphism, Micro-animations)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek Main Background */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
    }
    
    /* Hide default Streamlit decorations */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Header styling with neon gradient text */
    .app-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic card styling for questions */
    .glass-card {
        background: rgba(22, 28, 45, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }
    
    /* Glassmorphic panel for scores & status */
    .status-panel {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    /* Styling option elements */
    .option-label {
        font-weight: 600;
        color: #38bdf8;
    }
    
    /* Confetti-style success screen */
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    
    /* Custom button styling overrides */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper function to find best checkpoint automatically
def find_checkpoint(model_type):
    fixed_name = f"checkpoints/best-neural-cat-{model_type}.ckpt"
    if os.path.exists(fixed_name):
        return fixed_name
        
    if model_type == "optimized":
        ckpts = glob.glob("checkpoints/best-neural-cat-optimized-*.ckpt")
    else:
        ckpts = glob.glob("checkpoints/best-neural-cat-[0-9]*.ckpt")
        
    if not ckpts:
        # Check standard checkpoints folder
        ckpts = glob.glob("checkpoints/*.ckpt")
        if not ckpts:
            return None
            
    # Return path with lowest val_loss
    best_ckpt = None
    min_loss = float('inf')
    for ckpt in ckpts:
        try:
            if "val_loss=" in ckpt:
                loss_part = ckpt.split("val_loss=")[-1].replace(".ckpt", "")
                if "_metrics" in loss_part:
                    loss_part = loss_part.split("_metrics")[0]
                loss_val = float(loss_part)
                if loss_val < min_loss:
                    min_loss = loss_val
                    best_ckpt = ckpt
        except Exception:
            pass
            
    return best_ckpt if best_ckpt else ckpts[0]

# --- Initialize Core Services (Domain & Use Cases) ---
@st.cache_resource
def get_services():
    question_repo = SQLAlchemyQuestionRepository()
    cat_model = NeuralCATModelAdapter()
    start_use_case = StartCATSessionUseCase(question_repo, cat_model)
    submit_use_case = SubmitAnswerUseCase(cat_model)
    return start_use_case, submit_use_case

start_use_case, submit_use_case = get_services()

# --- Sidebar Configuration (Adaptive Test Parameters) ---
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình Tham số CAT")
    
    model_type = st.selectbox(
        "Phiên bản Mô hình (Model Version)",
        options=["optimized", "base"],
        index=0,
        help="optimized: Tích hợp embeddings câu hỏi và các đặc trưng ngôn ngữ/misconception.\nbase: Chỉ dùng embeddings câu hỏi."
    )
    
    # Checkpoint selection
    detected_ckpt = find_checkpoint(model_type)
    default_ckpt_path = detected_ckpt if detected_ckpt else ""
    
    checkpoint_path = st.text_input(
        "Đường dẫn Checkpoint (.ckpt)",
        value=default_ckpt_path,
        help="Nhập đường dẫn đến file checkpoint đã huấn luyện của mô hình."
    )
    
    st.markdown("---")
    st.markdown("#### 🎯 Luật Dừng Bài Thi (Stopping Rules)")
    
    se_threshold = st.slider(
        "Ngưỡng Sai số Chuẩn (SE threshold)",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        help="Khi sai số chuẩn ước lượng điểm năng lực của học sinh giảm xuống dưới ngưỡng này, bài thi sẽ tự động dừng."
    )
    
    min_questions = st.slider(
        "Số câu hỏi tối thiểu (MIN)",
        min_value=3,
        max_value=15,
        value=5,
        step=1
    )
    
    max_questions = st.slider(
        "Số câu hỏi tối đa (MAX)",
        min_value=10,
        max_value=50,
        value=30,
        step=1
    )
    
    st.markdown("---")
    st.markdown("#### 🧭 Thuật toán Chọn Câu (Item Selection)")
    
    selection_method = st.selectbox(
        "Phương pháp tổ hợp (Aggregation Method)",
        options=["multiplication", "addition"],
        index=0,
        help="multiplication: Nhân lượng thông tin Fisher với mức độ bổ trợ kỹ năng (ràng buộc cứng).\naddition: Tổ hợp tuyến tính (bù trừ)."
    )
    
    st.markdown("##### Siêu tham số Sigmoid Routing")
    lambda_max = st.slider("Trần Khám phá (λ max)", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
    lambda_min = st.slider("Đáy Khai thác (λ min)", min_value=0.0, max_value=0.4, value=0.1, step=0.05)
    beta_val = st.slider("Hệ số độ dốc (Beta)", min_value=0.1, max_value=1.5, value=0.5, step=0.05)
    k_pivot = st.slider("Điểm xoay trục (Pivot k)", min_value=5.0, max_value=25.0, value=12.0, step=1.0)
    
    st.markdown("---")
    
    # Initialize or reset test session
    if st.button("🚀 KHỞI TẠO BÀI THI MỚI", use_container_width=True):
        st.session_state.clear()
        st.session_state.start_test = True

# --- App Title & Layout ---
st.markdown("<div class='app-header'>NEURAL-CAT TESTING DEMO</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Hệ thống khảo thí thích ứng thông minh dựa trên mô hình Deep Learning NeuralCAT (4PL)</div>", unsafe_allow_html=True)

# Define initial session states
if "session_entity" not in st.session_state:
    st.session_state.session_entity = None
    st.session_state.current_question = None
    st.session_state.questions_bank = None
    
    # Track student interactive choices
    st.session_state.history_log = []  # List of dicts tracking step-by-step metadata
    st.session_state.info_history = []  # List of floats for Fisher Info values
    st.session_state.start_time = 0.0  # Timestamp when current question is displayed
    st.session_state.selected_option = None

# --- Trigger session start ---
if "start_test" in st.session_state and st.session_state.start_test:
    st.session_state.start_test = False
    
    # Load settings from UI inputs
    try:
        with st.spinner("Đang tải dữ liệu và khởi tạo phiên thi thích ứng... (Vui lòng đợi vài giây)"):
            sess, first_question, q_bank = start_use_case.execute(
                user_id=1,  # Simulation user ID
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                max_questions=max_questions,
                min_questions=min_questions,
                se_threshold=se_threshold,
                lambda_max=lambda_max,
                lambda_min=lambda_min,
                beta=beta_val,
                k_pivot=k_pivot,
                selection_method=selection_method
            )
            
            # Save initialized objects back to Session State
            st.session_state.session_entity = sess
            st.session_state.current_question = first_question
            st.session_state.questions_bank = q_bank
            
            st.session_state.history_log = []
            st.session_state.info_history = []
            # Append step 0 stats: initial SE is 1.0, mastery is mean of theta_0
            st.session_state.start_time = 0.0
            
            st.rerun()
    except Exception as e:
        st.error(f"Lỗi khởi chạy phiên thi: {e}")
        st.info("Gợi ý: Vui lòng kiểm tra lại đường dẫn checkpoint hoặc các cấu hình cơ sở dữ liệu.")

# --- Luồng xử lý Bắt đầu bài thi ---
if st.session_state.session_entity is None:
    st.markdown(
        """
        <div class='glass-card' style='text-align: center;'>
            <h2>Chào mừng bạn đến với Demo Khảo thí Thích ứng</h2>
            <p style='color:#94a3b8; font-size:1.1rem; max-width:600px; margin: 0 auto 2rem auto;'>
                Hệ thống sử dụng mô hình học sâu NeuralCAT (tích hợp 4PL IRT) để ước lượng mức độ làm chủ của học sinh trên 1,175 khái niệm kiến thức (knowledge concepts) và lựa chọn câu hỏi tiếp theo một cách tối ưu từ ngân hàng đề kiểm thử.
            </p>
            <p style='color:#38bdf8; font-weight:600;'>Vui lòng điều chỉnh các cấu hình ở sidebar bên trái và nhấn nút "Khởi tạo bài thi mới" để bắt đầu.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# --- Load session states to short variables ---
session = st.session_state.session_entity
current_q = st.session_state.current_question
q_bank = st.session_state.questions_bank

# Setup start time of the question if not set
if st.session_state.start_time == 0.0:
    st.session_state.start_time = time.time()

# --- Main Workspace Split: Left for Questions, Right for Stats & Chart ---
left_col, right_col = st.columns([5, 4], gap="large")

with left_col:
    if session.is_completed:
        st.markdown(
            f"""
            <div class='success-box'>
                <h1 style='color: #10b981; font-size: 2.2rem;'>🎉 BÀI THI HOÀN THÀNH!</h1>
                <p style='color: #94a3b8; margin-bottom: 2rem;'>Hệ thống đã đạt được điều kiện dừng thích ứng. Năng lực của thí sinh đã được định vị thành công.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        assert current_q is not None
        # Progress status
        t_current = len(session.responses) + 1
        st.markdown(f"#### 📝 Câu hỏi số {t_current} / tối đa {session.max_questions}")
        st.progress(min(1.0, (t_current - 1) / session.max_questions))
        
        # Display the question inside glassmorphic card
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"##### **Nội dung câu hỏi:**")
        st.markdown(current_q["content"])
        
        # Display Concepts of the Question
        concepts = current_q["concept_ids"] or []
        concepts_str = ", ".join([f"Concept {c}" for c in concepts])
        st.markdown(f"<p style='color:#64748b; font-size:0.9rem;'>📚 Khái niệm kỹ năng: <b>{concepts_str}</b></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Answer Options Selector
        st.markdown("##### **Chọn câu trả lời của bạn:**")
        
        options_dict = current_q["options"] or {}
        correct_answers = current_q["answer"] or []
        
        # Format option options for presentation
        if options_dict:
            # We present choices A, B, C, D...
            option_keys = sorted(list(options_dict.keys()))
            option_choices = [f"{key}. {options_dict[key]}" for key in option_keys]
            
            # Show Radio selector
            user_choice = st.radio(
                "Lựa chọn đáp án:",
                options=option_choices,
                index=None,
                label_visibility="collapsed"
            )
            
            if user_choice:
                # Extract key from string (e.g. "A. Option content" -> "A")
                st.session_state.selected_option = user_choice.split(".")[0]
        else:
            # If no options in database, fallback to binary Correct / Incorrect simulator
            st.markdown(
                """
                > [!NOTE]
                > Câu hỏi này không chứa các phương án trắc nghiệm sẵn trong cơ sở dữ liệu. Bạn hãy giả lập bằng cách chọn câu trả lời Đúng hay Sai trực tiếp:
                """
            )
            simulated_response = st.selectbox(
                "Mô phỏng kết quả làm bài:",
                options=["Đúng (Correct)", "Sai (Incorrect)"],
                index=None
            )
            if simulated_response:
                st.session_state.selected_option = "CORRECT" if "Đúng" in simulated_response else "INCORRECT"
                
        # Submit Button
        submit_clicked = st.button("🚀 NỘP CÂU TRẢ LỜI")
        
        if submit_clicked:
            if st.session_state.selected_option is None:
                st.warning("Vui lòng chọn một đáp án trước khi nộp bài.")
            else:
                # Calculate response time
                end_time = time.time()
                elapsed_time = max(2.0, end_time - st.session_state.start_time)
                
                # Determine correctness
                is_correct = 0
                if options_dict:
                    # Check if selected option key matches any correct answers
                    is_correct = 1 if st.session_state.selected_option in correct_answers else 0
                else:
                    is_correct = 1 if st.session_state.selected_option == "CORRECT" else 0
                    
                # Save details for the history report before calling usecase which changes session
                q_text = current_q["content"]
                q_id = current_q["id"]
                user_ans_label = st.session_state.selected_option
                correct_ans_label = ", ".join(correct_answers) if correct_answers else "Đúng"
                analysis_text = current_q.get("analysis", "")
                
                # Execute Submit Use Case
                updated_session, next_q, current_se = submit_use_case.execute(
                    session=session,
                    questions_bank=q_bank,
                    question_id=q_id,
                    is_correct=is_correct,
                    response_time=elapsed_time,
                    info_history=st.session_state.info_history
                )
                
                # Update Session State Entity
                st.session_state.session_entity = updated_session
                st.session_state.current_question = next_q
                
                # Log step interaction detail
                st.session_state.history_log.append({
                    "step": t_current,
                    "question_id": q_id,
                    "question_text": q_text,
                    "user_answer": user_ans_label,
                    "correct_answer": correct_ans_label,
                    "is_correct": is_correct,
                    "time_sec": elapsed_time,
                    "se": current_se,
                    "theta": float(updated_session.mastery_history[-1].mean()) * 100,  # Mean mastery in percentage
                    "analysis": analysis_text
                })
                
                # Reset display variables for next round
                st.session_state.start_time = 0.0
                st.session_state.selected_option = None
                
                # Rerun Streamlit to refresh the UI
                st.rerun()

# --- Right Workspace Split: Live Dashboard Stats & Chart ---
with right_col:
    st.markdown("### 📊 Tiến trình Đánh giá Năng lực")
    
    # Calculate stats
    t_done = len(session.responses)
    accuracy = (sum(session.responses) / t_done * 100) if t_done > 0 else 0.0
    latest_se = session.se_history[-1] if session.se_history else 1.0
    
    # Fetch student ability estimates
    # mastery_history contains vectors of size K. We take the mean to show generalized ability
    # In CAT, ability theta_t is between 0 and 1 (Sigmoid Mastery). We show as percentage
    latest_mastery = float(session.mastery_history[-1].mean()) * 100 if session.mastery_history else 50.0
    
    # Render indicators panel
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class='status-panel'>
                <p style='margin:0; font-size:0.85rem; color:#94a3b8; font-weight:600;'>ĐIỂM NĂNG LỰC (θ)</p>
                <h2 style='margin:0; color:#38bdf8;'>{latest_mastery:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class='status-panel' style='border-left-color: #f59e0b;'>
                <p style='margin:0; font-size:0.85rem; color:#94a3b8; font-weight:600;'>SAI SỐ CHUẨN (SE)</p>
                <h2 style='margin:0; color:#f59e0b;'>{latest_se:.3f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class='status-panel' style='border-left-color: #10b981;'>
                <p style='margin:0; font-size:0.85rem; color:#94a3b8; font-weight:600;'>ĐỘ CHÍNH XÁC</p>
                <h2 style='margin:0; color:#10b981;'>{accuracy:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # --- Dual Y-Axis Line Chart using Plotly ---
    steps = list(range(len(session.se_history)))
    
    # generalized mastery vector
    mastery_curve = [float(m.mean()) * 100 for m in session.mastery_history]
    se_curve = session.se_history
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Mastery Line
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mastery_curve,
            name="Điểm Năng lực (θ)",
            line=dict(color="#38bdf8", width=3),
            mode="lines+markers",
            marker=dict(size=8)
        ),
        secondary_y=False
    )
    
    # Add Standard Error Line
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=se_curve,
            name="Sai số Chuẩn (SE)",
            line=dict(color="#f59e0b", width=3, dash="dash"),
            mode="lines+markers",
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    # Styling Plotly Chart in Dark Theme
    fig.update_layout(
        title="<b>Biến động Năng lực & Sai số Chuẩn qua từng bước</b>",
        title_font_size=16,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Thứ tự câu hỏi đã làm (Câu)",
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickmode="linear",
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            title="Năng lực trung bình (θ %)",
            gridcolor="rgba(255, 255, 255, 0.05)",
            range=[0, 100],
            titlefont=dict(color="#38bdf8"),
            tickfont=dict(color="#38bdf8")
        ),
        yaxis2=dict(
            title="Sai số Chuẩn (SE)",
            range=[0, 1.05],
            titlefont=dict(color="#f59e0b"),
            tickfont=dict(color="#f59e0b")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- Post-Test Detailed Evaluation Report ---
if session.is_completed and st.session_state.history_log:
    st.markdown("---")
    st.markdown("### 📋 Báo cáo Chi tiết Phiên thi Thích ứng")
    
    # Summary of metrics
    final_m = mastery_curve[-1]
    final_se = se_curve[-1]
    
    st.markdown(
        f"""
        Ở mốc câu thứ **{t_done}**, sai số chuẩn đã giảm xuống mức **{final_se:.3f}** (hội tụ dưới ngưỡng **{session.se_threshold}** hoặc đã đạt giới hạn câu hỏi tối đa).
        Năng lực cuối cùng của bạn đạt **{final_m:.1f}%** trên tổng số 1,175 khái niệm kiểm tra.
        """
    )
    
    # Detailed log table
    for log in st.session_state.history_log:
        status_icon = "✅" if log["is_correct"] == 1 else "❌"
        status_color = "#10b981" if log["is_correct"] == 1 else "#ef4444"
        
        with st.expander(f"Câu {log['step']}: {status_icon} ID: {log['question_id']} | Thời gian: {log['time_sec']:.1f}s | Năng lực (θ): {log['theta']:.1f}%"):
            st.markdown(f"**Nội dung câu hỏi:**")
            st.markdown(log["question_text"])
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Đáp án của bạn:** `{log['user_answer']}`")
            with col_b:
                st.markdown(f"**Đáp án đúng:** `{log['correct_answer']}`")
                
            st.markdown(f"**Sai số SE sau bước:** `{log['se']:.3f}`")
            
            if log["analysis"]:
                st.markdown(f"**Phân tích & Giải thích:**")
                st.info(log["analysis"])

