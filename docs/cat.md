# Computerized Adaptive Testing Engine

# **CƠ CHẾ ĐỊNH TUYẾN VÀ ĐIỀU KIỆN DỪNG TRONG NEURAL CAT**

# **1. Điều kiện dừng (Stopping Rules)**

Quá trình lặp đưa ra câu hỏi mới sẽ dừng khi hệ thống thỏa mãn một trong ba điều kiện sau:

## **1.1. Điều kiện giới hạn chiều dài bài thi (Khóa trên)**

Để ngăn chặn tình trạng bài thi kéo dài quá mức gây mệt mỏi cho thí sinh, hệ thống thiết lập một ngưỡng ngắt dựa trên tổng số câu hỏi. Quá trình kiểm tra sẽ tự động dừng nếu số lượng câu hỏi đã phản hồi đạt giới hạn tối đa:

$$
\text{num\_question} \ge MAX_{question}
$$

## **1.2. Điều kiện hội tụ Sai số Đo lường (Tối ưu hóa cốt lõi)**

Đây là tiêu chí dừng mang tính đặc thù của các hệ thống Khảo thí Thích ứng (CAT), nhằm đảm bảo năng lực tiềm ẩn của cá nhân đã được ước lượng với độ tin cậy đủ lớn.

Khác với mô hình IRT 1PL/2PL truyền thống sử dụng tổng Lượng thông tin Fisher để tính sai số, đối với kiến trúc Neural CAT tích hợp 4PL (có sự biến thiên phức tạp của độ đoán mò $g$ và độ bất cẩn $s$), hệ thống sử dụng **Phương sai Hậu nghiệm (Posterior Variance)** của trạng thái năng lực ẩn làm thước đo Sai số Chuẩn ($SE_t$).

$$
SE_t = \sqrt{\text{Var}(\theta_t | X_t)}
$$

Trong đó, $\text{Var}(\theta_t | X_t)$ là phương sai của phân phối năng lực tại bước $t$ được chiết xuất từ mạng nơ-ron cập nhật trạng thái. Hệ thống sẽ kích hoạt lệnh dừng khi thỏa mãn đồng thời hai yếu tố: chiều dài bài thi đạt mức tối thiểu ($MIN_{question}$) và sai số đo lường đã hội tụ dưới ngưỡng cho phép ($\epsilon_{SE}$):

$$
(\text{num\_question} \ge MIN_{question}) \wedge (SE_t < \epsilon_{SE})
$$

### **1.3. Điều kiện giới hạn thời gian**

Để đảm bảo tính chuẩn hóa về mặt quản trị thời gian đối với mọi phiên thi, thuật toán tích hợp một điều kiện ngắt khẩn cấp dựa trên đồng hồ đếm ngược. Trạng thái kết thúc sẽ được ghi nhận khi:

$$
\text{time\_out} = \text{True}
$$

# Item Selection

Trong quá trình quản lý bài kiểm tra thích ứng, bài toán lựa chọn câu hỏi kế tiếp được mô hình hóa dưới dạng một quá trình tối ưu hóa đa mục tiêu. Mục tiêu cốt lõi là tìm ra câu hỏi $j^*$ từ ngân hàng câu hỏi sao cho tối đa hóa một hàm điểm số tổng hợp $Score(j)$:

$$
j^* = \arg\max_{j} Score(j)
$$

Hàm mục tiêu $Score(j)$ được xây dựng dựa trên sự kết hợp của hai thành phần độc lập:

- Hàm Lượng thông tin Fisher (Fisher Information) mở rộng cho 4PL:
    
    $$
    \text{Info}(j) = \frac{\big[ (1 - s^j - g^j) \cdot K^j \cdot (1 - K^j) \big]^2}{P_{t+1}^j \cdot (1 - P_{t+1}^j)}
    $$
    
- **Giá trị Bổ trợ $\text{Bonus}(j)$:** Đại diện cho mức độ đáp ứng các ràng buộc về nội dung và cấu trúc bài thi (content balancing), được định lượng bằng tổng trọng số của các kỹ năng mà câu hỏi cung cấp:
    
    $$
    \text{Bonus}(j) \propto \sum_{k=1}^{K} c_{j,k} \cdot w_k \cdot f_k
    $$
    

Để tích hợp hai thành phần này, hệ thống có thể áp dụng một trong hai phương pháp tiếp cận sau:

**1. Phương pháp Cộng Tuyến tính**

Phương pháp này xây dựng hàm mục tiêu dưới dạng một tổ hợp tuyến tính của thành phần Thông tin và thành phần Bổ trợ:

$$
Score(j) = (1 - \lambda) \cdot \text{Info}(j) + \lambda \cdot \text{Bonus}(j)
$$

**Cơ chế hoạt động và Triết lý:**

Mô hình này vận hành theo triết lý **bù trừ**. Cụ thể, sự thiếu hụt ở một tiêu chí có thể được bù đắp bằng sự vượt trội ở tiêu chí còn lại.

- **Vai trò của siêu tham số $\lambda$:** $\lambda$ (nhận giá trị trong đoạn $[0, 1]$) đóng vai trò là tham số điều hướng, cho phép kiểm soát sự đánh đổi giữa độ chính xác đo lường và mức độ bao phủ nội dung.
    - Khi $\lambda$ tiến gần về $1$ (ví dụ: $\lambda = 0.8$), hệ thống đặt trọng tâm vào việc thỏa mãn cấu trúc nội dung bài thi, xem nhẹ tạm thời mức độ phù hợp về độ khó.
    - Khi $\lambda$ tiến gần về $0$ (ví dụ: $\lambda = 0.2$), hệ thống ưu tiên tối đa hóa hàm lượng thông tin thống kê để thu hẹp sai số đo lường, chấp nhận sự sai lệch nhỏ trong phân phối nội dung.
- **Hạn chế lý thuyết (Sự nhượng bộ rủi ro):** Điểm yếu cốt lõi của phương pháp này nằm ở rủi ro phá vỡ cấu trúc bài thi do tính chất bù trừ. Giả sử một nhóm kỹ năng đã được đánh giá đầy đủ (làm cho $f_k = 0$, suy ra $\text{Bonus} = 0$). Nếu ngân hàng tồn tại một câu hỏi thuộc nhóm kỹ năng này có độ khó trùng khớp hoàn hảo với năng lực thí sinh ($\text{Info} \approx 1.0$), hàm mục tiêu vẫn trả về một giá trị dương $Score(j) = (1 - \lambda) \cdot 1.0 > 0$. Do đó, thuật toán **vẫn có khả năng** lựa chọn câu hỏi này, dẫn đến hiện tượng kiểm tra thừa đối với một số kỹ năng nhất định.

**2. Phương pháp Nhân**

Thay vì sử dụng phép cộng, phương pháp này tích hợp hai thành phần thông qua tích số:

$$
Score(j) = \text{Info}(j) \cdot \text{Bonus}(j)
$$

**Cơ chế hoạt động và Triết lý:**

Mô hình này vận hành theo nguyên tắc **phi bù trừ**, tạo ra một cơ chế **"phủ quyết cứng"**. Sự ràng buộc giữa đo lường và nội dung là tuyệt đối.

- **Cơ chế tự điều chỉnh trọng số:** Phương pháp này loại bỏ hoàn toàn sự phụ thuộc vào tham số $\lambda$ ngoại sinh. Hệ thống tự động nội suy trọng số, trong đó cả hai điều kiện (độ khó và nội dung) đều mang tính tiên quyết và có tầm quan trọng ngang hàng nhau.
- **Ưu điểm vượt trội về tuân thủ cấu trúc:** Đặc tính quan trọng nhất của phép nhân là khả năng triệt tiêu giá trị hàm mục tiêu. Nếu một câu hỏi chỉ chứa các kỹ năng đã đạt đủ số lượng yêu cầu ($\text{Bonus} = 0$), thì toàn bộ hàm $Score(j)$ lập tức nhận giá trị $0$. Bất chấp câu hỏi đó có độ khó lý tưởng đến đâu (xác suất $P = 0.5$ mang lại $\text{Info}$ tối đa), nó vẫn bị loại bỏ hoàn toàn khỏi không gian lựa chọn. Cơ chế này đảm bảo hệ thống **tuân thủ tuyệt đối và không bao giờ vi phạm** các thiết lập về cấu trúc kỹ năng của bài thi.

# Mối quan hệ giữa k và $\lambda$

## Mô hình Sigmoid (Logistic Function)

Thay vì đợi đến đúng câu $k$ mới bắt đầu đổi chiến thuật, hàm Sigmoid sẽ tạo ra một đường cong hình chữ S úp ngược. Hệ thống sẽ có xu hướng "chuẩn bị chuyển giao" từ vài câu trước $k$, và hoàn tất việc giảm $\lambda$ ở vài câu sau $k$.

$$
\lambda_t = \lambda_{min} + \frac{\lambda_{max} - \lambda_{min}}{1 + e^{\beta(t - k)}}
$$

Trong đó:

**$\lambda_{max}$ (Trần Khám phá):**

- *Ý nghĩa toán học:* Là giới hạn trên cùng của hàm số. Khi $t$ rất nhỏ (mới bắt đầu bài thi), $\lambda_t \approx \lambda_{max}$.
- *Ý nghĩa thực tế:* Xác định mức độ **ưu tiên tối đa cho việc "quét" đủ kỹ năng**. Nếu bạn đặt $\lambda_{max} = 0.8$, nghĩa là hệ thống dồn 80% sức mạnh để đi tìm những chủ đề chưa hỏi, và chỉ chừa 20% quan tâm đến độ khó của câu đó. (Không nên đặt 1.0 vì hệ thống sẽ chọn bừa câu hỏi cực khó/cực dễ miễn là đúng chủ đề).

**$\lambda_{min}$ (Đáy Khai thác):**

- *Ý nghĩa toán học:* Là giới hạn dưới cùng của hàm số. Khi $t$ rất lớn (cuối bài thi), $\lambda_t \approx \lambda_{min}$.
- *Ý nghĩa thực tế:* Xác định mức độ **nhượng bộ tối thiểu cho cấu trúc kỹ năng**. Nếu $\lambda_{min} = 0.2$, ở cuối bài thi, hệ thống dồn 80% sức mạnh để tìm câu hỏi có độ khó khớp xác suất $P=0.5$ (để chốt điểm), và chỉ ưu tiên 20% cho việc hỏi đúng cấu trúc.

**$k$ (Điểm xoay trục / Ngưỡng hội tụ):**

- *Ý nghĩa toán học:* Đây là trung điểm của đường cong (tâm đối xứng). Tại đúng vị trí $t = k$, phần lũy thừa $e^{\beta(k-k)} = e^0 = 1$. Lúc này $\lambda_k$ nằm chính giữa $\lambda_{max}$ và $\lambda_{min}$.
- *Ý nghĩa thực tế:* Đây là **thời điểm vàng** mà hệ thống chuyển từ trạng thái "Thu thập dữ liệu diện rộng" sang "Đo lường năng lực chuyên sâu". Tại câu $k$, chiến thuật là cân bằng 50/50.
- *Ví dụ:* $k = 15$ nghĩa là hệ thống kỳ vọng sau 15 câu thì sẽ nắm được tương đối năng lực của thí sinh.

**$\beta$ (Hệ số độ dốc):**

- *Ý nghĩa toán học:* Quyết định tốc độ tiến về $0$ của phân số khi $t$ vượt qua $k$.
- *Ý nghĩa thực tế:* Đây là **tốc độ "bẻ lái"** của hệ thống.
    - Nếu $\beta$ lớn (ví dụ $1.5$): Đường cong lao xuống như vách đá. Bài thi chuyển từ Khám phá sang Khai thác cực kỳ gắt gao ngay tại quanh mốc $k$.
    - Nếu $\beta$ nhỏ (ví dụ $0.2$): Đường cong lài như đồi dốc. Sự chuyển giao diễn ra từ từ, êm ái qua rất nhiều câu hỏi.

**$t$ (Thứ tự câu hỏi hiện tại):**

- Trục hoành của đồ thị. Đây là thông số duy nhất liên tục thay đổi sau mỗi lần thí sinh submit một câu trả lời. Sự chênh lệch $(t - k)$ chính là "khoảng cách" từ hiện tại đến điểm xoay trục, báo hiệu cho hệ thống biết đang ở Pha Khám phá ($t < k$), Pha Chuyển giao ($t \approx k$), hay Pha Khai thác ($t > k$).

## **Trường phái Thống kê (Dựa trên M-IRT & Độ bất định)**

Mỗi thí sinh có một quỹ đạo bộc lộ năng lực khác nhau:

- **Trường hợp Thí sinh "Nhất quán":** Có những người kiến thức rất rõ ràng (vững toàn diện hoặc hổng toàn diện). Quỹ đạo năng lực ước lượng ($\Theta_t$) của họ hội tụ cực kỳ nhanh chỉ sau 5-7 câu. Nếu giữ cố định $k = 15$, hệ thống sẽ lãng phí 8-10 câu hỏi tiếp theo để "khám phá" những thứ đã quá rõ ràng. Lúc này, $k$ nên được hệ thống **tự động co ngắn lại** để chốt điểm sớm.
- **Trường hợp Thí sinh "Bất ổn" (Nhiễu):** Có thí sinh đánh lụi, hoặc "học tài thi phận" (đúng câu siêu khó nhưng sai câu cực dễ). Trạng thái $\Theta_t$ dao động liên tục. Nếu $k = 15$ là quá ngắn đối với họ, hệ thống sẽ bị ép chuyển sang chế độ "đo lường gắt gao" ($\lambda$ thấp) khi chưa đủ tự tin, dẫn đến đánh giá sai lệch. Lúc này, $k$ phải **tự động giãn ra** (thêm thời gian khám phá) và $\beta$ cần **nhỏ lại** để chuyển giao chiến thuật thật chậm.

Thay vì dùng $k$ như một biến đếm thời gian (số lượng câu hỏi), hãy dùng **Sai số chuẩn (Standard Error - SE)** hoặc **Biến thiên (Variance)** của mức năng lực ước lượng làm thước đo.

- **Cơ chế:** Sau mỗi câu hỏi $t$, hệ thống tính phương sai của ước lượng năng lực $\Theta_t$.
- **Điểm xoay trục:** $k$ không còn là một con số nguyên (ví dụ: câu thứ 15), mà được định nghĩa bằng một **Ngưỡng hội tụ ($\epsilon$)**. Khi phương sai giảm xuống dưới mức $\epsilon$ (tức là hệ thống tin rằng: *"Tôi đã đoán khá chính xác năng lực người này rồi"*), thuật toán lập tức kích hoạt đường cong Sigmoid để hạ $\lambda$ xuống. Quá trình này có thể xảy ra ở câu thứ 8 đối với thí sinh A, nhưng đến tận câu 22 mới xảy ra đối với thí sinh B.