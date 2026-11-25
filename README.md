# NIDS-ML

Đồ án **xây dựng hệ thống phát hiện xâm nhập mạng sử dụng học máy** (Network Intrusion Detection System). Mô hình phân loại network flow thành hai nhóm: **Normal** (bình thường) và **Attack** (tấn công).

## Dữ liệu và phương pháp

- **Dataset:** CICIDS2017, gồm 8 file CSV chứa các đặc trưng thống kê network flow.
- **Tiền xử lý:** kiểm tra nhãn, NaN/Infinity, giá trị âm và bản ghi trùng; giữ thông tin ngày, file nguồn và nhãn gốc phục vụ đánh giá.
- **Dữ liệu sau xử lý:** 2.572.640 dòng, 82 cột, gồm 78 feature đầu vào và 4 cột nhãn/metadata.
- **Mô hình:** Logistic Regression, Decision Tree, Random Forest, XGBoost và MLP.
- **Chỉ số đánh giá:** Accuracy, Precision, Recall Attack, F1, Macro-F1, FPR, FNR và confusion matrix.

Hai phương pháp chia dữ liệu được sử dụng để so sánh hiệu quả:

| Phương pháp | Cách chia | Train | Test |
|---|---|---:|---:|
| **P1 — Random Group-Aware** | Chia theo nhóm feature, tránh để các vector feature giống nhau xuất hiện ở cả Train và Test | 1.800.661 | 771.979 |
| **P2 — Day-Disjoint** | Train Monday–Wednesday, Test Thursday–Friday; loại khỏi Train các vector feature trùng Test | 1.506.387 | 1.037.872 |

Mỗi thuật toán được chọn cấu hình trên Validation nội bộ, sau đó fit lại trên toàn Train tương ứng và đánh giá Test. Các bước học tham số như scaling chỉ được fit trên dữ liệu huấn luyện. Validation P2 là chia theo nhóm trong Monday–Wednesday, không phải tập giữ riêng theo ngày.

## Kết quả hiện tại

Đã hoàn tất **10 mô hình ứng viên**: 5 thuật toán cho mỗi phương pháp P1/P2. Bảng dưới tổng hợp kết quả Test, đơn vị **%**:

| Mô hình | P1 Recall Attack | P1 Macro-F1 | P2 Recall Attack | P2 Macro-F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 58,21 | 82,56 | 36,56 | 71,73 |
| Decision Tree | 99,77 | 99,78 | 37,96 | 73,57 |
| Random Forest | 99,26 | 99,70 | 36,43 | 72,69 |
| XGBoost | 99,83 | 99,86 | 36,41 | 72,67 |
| MLP | 98,79 | 99,45 | 19,58 | 61,29 |

Hiệu quả phát hiện Attack giảm rõ rệt trên P2. Do cả ngày thu thập và họ tấn công cùng thay đổi giữa Train/Test, không thể quy toàn bộ chênh lệch cho yếu tố thời gian.

## Cấu trúc project

```text
NIDS-ML/
├── Notebooks/
│   ├── data_exploration.ipynb       # Khám phá dữ liệu
│   ├── data_preprocessing.ipynb     # Làm sạch và audit dữ liệu
│   ├── p1_random_group_aware.ipynb  # Thực nghiệm P1
│   ├── p2_day_disjoint.ipynb        # Thực nghiệm P2
│   └── *.py                       # Helper chia tập, huấn luyện, đánh giá
├── Dataset/
│   ├── CIC-IDS-2017/               # CSV gốc, chỉ lưu cục bộ
│   └── Processed/                 # Dữ liệu xử lý và kết quả thực nghiệm
├── .gitignore
└── README.md
```

Repository lưu code, notebook với output đã chạy, JSON kết quả/manifest và CSV audit nhỏ. Dataset gốc, dữ liệu xử lý dạng gzip, model đã train và cache được bỏ qua bởi `.gitignore`; cần chuẩn bị riêng nếu muốn chạy lại thí nghiệm.

**Công nghệ chính:** Python, Jupyter Notebook, pandas, NumPy, scikit-learn, XGBoost và joblib. Thứ tự đọc notebook: khám phá dữ liệu → tiền xử lý → P1/P2.