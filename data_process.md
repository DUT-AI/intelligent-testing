[Dữ liệu thô ban đầu]
│
▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: Chuẩn hóa & Cắt Đoạn (Sessionization)    │
│ - Chuẩn hóa mốc thời gian tuyệt đối sang datetime       │
│ - Cắt session ở ngưỡng 60 phút (Tránh nát chuỗi)       │
│ - Sinh đặc trưng động: response_time (Đơn vị: Giây)    │
│ - Giữ lại hành vi Đánh lụi nhanh (Fast Guessing)       │
└────────────────────────────────────────────────────────┘
│
▼
[DataFrame Session Đã Làm Sạch]
│
├────────────────────────────────────────┐
▼                                        ▼
┌──────────────────────────────────────┐   ┌───────────────────────────┐
│ GIAI ĐOẠN 2: Chia Tập Train/Val      │   │ TẬP KIỂM THỬ (TEST)       │
│ - Tổng hợp hồ sơ năng lực theo UID   │   │ - Giữ nguyên phân phối     │
│ - Phân tầng User (Stratification)    │   │ - Gán nhãn 'test'         │
│ - Chia tách 85% Train / 15% Val      │   │ - Xuất file kết quả       │
└──────────────────────────────────────┘   └───────────────────────────┘
│
▼
┌──────────────────────────────────────┐
│ ĐẦU RA CUỐI CÙNG                     │
│ - final_train.csv                    │
│ - final_val.csv                      │
│ - final_test.csv                     │
└──────────────────────────────────────