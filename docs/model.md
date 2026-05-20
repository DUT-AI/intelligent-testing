# **Kiến trúc Hệ thống Đánh giá Năng lực và Khảo thí Thích ứng (Neural CAT Engine)**

Kiến trúc đề xuất là một hệ thống Đánh giá Năng lực Thích ứng giải quyết bài toán cốt lõi của Khảo thí trên máy tính (CAT): Dự đoán chính xác trạng thái nhận thức đa chiều của người học và chống lại các nhiễu loạn tâm lý (đoán mò, bất cẩn). Mô hình tích hợp Lý thuyết Ứng đáp Câu hỏi 4 tham số (4PL IRT) vào kiến trúc Deep Learning, tạo thành một vòng lặp kín gồm 4 khối chức năng: Lọc nhiễu Đầu vào (Input Refiner), Biểu diễn Không gian (Embedding), Theo dõi Nhận thức (Knowledge Tracing), và Dự đoán - Định tuyến (Prediction & Routing).

## **1. Khối Lọc nhiễu và Tinh chỉnh Tương tác (4PL Input Refiner)**

**Mục đích:** Tại thời điểm $t$, hệ thống nhận kết quả trả lời thô $r_t \in \{0, 1\}$ của học sinh. Khối này đóng vai trò như một màng lọc XAI, sử dụng phương trình 4PL IRT để bóc tách xác suất học sinh "đoán mò" (Guessing) hoặc "bất cẩn" (Slips), từ đó chuyển nhãn thô thành "Tín nhiệm mềm" (Soft-label) $r'_t$.

**Quá trình xử lý:**
Dựa trên mạng nơ-ron chia sẻ trọng số (Shared MLP), hệ thống trích xuất 3 tham số IRT của câu hỏi $t$:

1. **Độ đoán mò ($g_t$) và Độ bất cẩn ($s_t$):** Chỉ phụ thuộc vào đặc trưng vật lý của câu hỏi.

$$
[g_{raw}, s_{raw}] = \text{MLP}_{item}(x_t)
$$

$$
g_t = \sigma(g_{raw}), \quad s_t = \sigma(s_{raw})
$$

1. Áp dụng luật Tín nhiệm mềm để hiệu chỉnh câu trả lời, đảm bảo tính liên tục cho đạo hàm:

$$
r'_t = \begin{cases} 1 - g_t, & \text{nếu } r_t = 1 \text{ (Giảm điểm do nghi ngờ đoán lụi)} \\ s_t, & \text{nếu } r_t = 0 \text{ (Cứu vớt điểm do phát hiện bất cẩn)} \end{cases}
$$

Giá trị $r'_t \in (0, 1)$ mang thông tin thuần khiết nhất về sự tiến bộ thực sự của học sinh.

## **2. Khối Biểu diễn Không gian Tương tác (Input Embedding Module)**

Mục đích của khối này là chuyển đổi các tương tác thô của học sinh thành các vector đặc trưng không gian, đồng thời tách biệt rõ ràng trạng thái "Thành thạo" (Làm đúng) và "Thiếu sót" (Làm sai).
Tại bước thời gian $t$, đầu vào bao gồm:

- Thời gian trả lời câu hỏi (Response Time): $T_t$
- Vector ngữ nghĩa của câu hỏi: $x_t \in \mathbb{R}^{d_v}$
- Kết quả trả lời thực tế của người dùng: $r'_t \in \{0, 1\}$

**Quá trình xử lý:**
Đầu tiên, hệ thống thực hiện chuẩn hóa thời gian trả lời bằng phép biến đổi logarit để giảm thiểu độ lệch chuẩn (skewness), sau đó đưa qua mạng nơ-ron truyền thẳng (MLP) để tạo vector nhúng thời gian $v_t$:

$$
v_t = \text{MLP}(\log(1 + T_t))
$$

hệ thống sử dụng phép **Cộng nội suy (Weighted Concatenation)** dựa trên nhãn mềm $r'_t$. Kỹ thuật này ép vector câu hỏi $x_t$ trượt mượt mà giữa hai không gian "Đúng" và "Sai":

$$
I'_t = r'_t \cdot [x_t \oplus v_t \oplus \mathbf{0}] + (1 - r'_t) \cdot [\mathbf{0} \oplus x_t \oplus v_t]
$$

(Với $\mathbf{0}$ là vector không có cùng số chiều $d_x + d_v$)

Cuối cùng, để mô hình nắm bắt được tính tuần tự của bài thi, thông tin vị trí (Positional Encoding) được cộng trực tiếp vào vector tương tác:

$$
I_t = I'_t + \text{Embedding}(pos_t, d_I)
$$

Đầu ra của khối là vector $I_t$ chứa toàn bộ đặc trưng ngữ cảnh của tương tác tại bước $t$.

## **3. Khối Mô hình hóa Chuỗi (Sequence Modeling Module)**

Khối này có nhiệm vụ tổng hợp lịch sử làm bài để rút trích ra các mẫu (patterns) hành vi và sự phát triển nhận thức của học sinh theo thời gian.

**Quá trình xử lý:**

Hệ thống tiếp nhận chuỗi các tương tác từ đầu bài thi đến thời điểm hiện tại: $X = \{I_1, I_2, \dots, I_t\}$

Chuỗi này được đưa qua cơ chế **Masked Multi-head Attention** (như trong kiến trúc Transformer). Cơ chế mặt nạ (Masking) đảm bảo tính nhân quả (causality), ngăn chặn mô hình truy cập rò rỉ thông tin từ các câu hỏi trong tương lai ($t+1$).

Đầu ra của khối là vector ngữ cảnh tổng hợp $h_t \in \mathbb{R}^{d_h}$, chứa đựng toàn bộ dữ liệu lịch sử tương tác được nén lại tính đến thời điểm $t$.

## **4. Khối Giải mã và Cập nhật Năng lực (Decoding Head)**

Đây là khối cốt lõi thực hiện nhiệm vụ đánh giá trạng thái nhận thức. Nó cập nhật ma trận năng lực $\theta$ dựa trên nguyên tắc: chỉ cập nhật những kiến thức liên quan đến câu hỏi hiện tại.

**Quá trình xử lý:**
Giả sử bài thi đánh giá $K$ kỹ năng, năng lực của học sinh được biểu diễn bằng ma trận $\theta_{t-1} \in \mathbb{R}^{K \times d_h}$. Tại $t=0$, $\theta_0$ được khởi tạo ngẫu nhiên hoặc khởi tạo bằng phân phối năng lực trung bình.

Dựa trên vector ngữ cảnh $h_t$, mạng nơ-ron dự đoán sự thay đổi (residual) của toàn bộ các kỹ năng:

$$
\Delta \hat{\theta}_t = \text{FFN}(h_t) \in \mathbb{R}^{K \times d_h}
$$

Để ngăn chặn hiện tượng cập nhật chéo không hợp lý (rò rỉ điểm), cơ chế **Cổng cứng (Hard Gating)** được áp dụng thông qua ma trận Q (Q-matrix) $Q_t$:

$$
\Delta \theta_t = Q_t \odot \Delta \hat{\theta}_t
$$

Trạng thái năng lực mới được cập nhật bằng cách cộng dồn sự thay đổi vào trạng thái cũ. Nhằm đảm bảo sự hội tụ điểm số ở cuối bài thi, hệ số hãm $\alpha_t$ (Damping Factor) được tích hợp dựa trên lịch trình Warm-up & Decay:

$$
\theta_t = \theta_{t-1} + \alpha_t \cdot \Delta \theta_t
$$

Với $\alpha_t$ được định nghĩa:

$$
\alpha_t = \begin{cases} \frac{t}{k} \cdot \alpha_{max}, & \text{nếu } t \le k \text{ (Giai đoạn Warm-up)} \\ \frac{1}{\sqrt{t - k + 1}} \cdot \alpha_{max}, & \text{nếu } t > k \text{ (Giai đoạn Hội tụ)} \end{cases}
$$

## **4. Khối Dự đoán (Hybrid 4PL Predictor)**

**Mục đích:** Dự đoán xác suất trả lời đúng $P$ cho các câu hỏi tiềm năng ($t+1$) trong ngân hàng đề, đồng thời tính toán Lượng thông tin Fisher làm tiêu chí định tuyến cho CAT.

**Quá trình xử lý:**

1. Khi xem xét một câu hỏi $j$ (tại thời điểm $t+1$), hệ thống phân bổ tỷ trọng yêu cầu của các kỹ năng dựa trên ma trận Q chuẩn hóa. Trọng số $\beta$ cho kỹ năng thứ $i$ được tính bằng:
    
    $$
    \beta_{t+1}^i = \frac{Q^{j, i}}{\sum_{k=1}^K Q^{j, k}}
    $$
    
2. **Độ chênh lệch năng lực ($\Delta_t$):** Giao thoa giữa năng lực học sinh $\theta_{t-1}$ và câu hỏi $x_t$.
    
    Tiếp theo, hệ thống trích xuất vector biểu diễn sức mạnh tổng hợp (Targeted Ability) của học sinh đối với riêng câu hỏi này bằng phép tổ hợp tuyến tính:
    
    $$
    S_{t+1}^j = \beta^j_{t+1} \theta_{t} \in \mathbb{R}^{d_h}
    $$
    
    $$
    \Delta_{t+1}^j = \text{MLP}_{\Delta}\left( S_{t+1}^j \oplus x_{t+1}^j \right)
    $$
    
    Trong đó, $\beta_{t+1} \in \mathbb{R}^{1 \times K}$ là vector trọng số phân bổ kỹ năng của câu hỏi tiềm năng $j$ tại thời điểm $t+1$, và $\theta_t \in \mathbb{R}^{K \times d_h}$ là ma trận năng lực hiện tại (vừa được cập nhật). Phép tổ hợp tạo ra sức mạnh mục tiêu $S_{t+1}^j$, sau đó được ghép nối với đặc trưng câu hỏi $x_{t+1}^j$ để tính ra Độ chênh lệch năng lực dự kiến $\Delta_{t+1}^j$.
    
3. Áp dụng phương trình 4PL IRT để tính **Xác suất trả lời đúng tổng hợp ($P$):**
    
    $$
    K^j = \frac{1}{1 + e^{-\Delta_{t+1}^j}}
    $$
    
    $$
    P_{t+1}^j = g^j + (1 - s^j - g^j) \cdot K^j
    $$
    
    (Trong đó $g^j, s^j$ được trích xuất từ câu hỏi $j$ qua item MLP).
    

# **Cấu trúc Huấn luyện Đa nhiệm (Multi-task Joint Training)**

Hệ thống được huấn luyện từ đầu đến cuối (End-to-End) bằng hàm mất mát tổng hợp nhằm tối ưu hóa đồng thời 2 mục tiêu: bám sát thực tế câu trả lời và diễn giải được thông số IRT.

$$
\mathcal{L}_{total} = \text{BCE}(P_{t+1}, r_{t+1}) + \lambda \cdot \mathcal{L}_{reg}(g, s)
$$

*(Trong đó BCE là Binary Cross-Entropy, $\mathcal{L}_{reg}$ là hàm chuẩn hóa ép các giá trị đoán mò/bất cẩn không được vọt lên quá vô lý, và $\lambda$ là siêu tham số cân bằng giữa 2 mục tiêu).*