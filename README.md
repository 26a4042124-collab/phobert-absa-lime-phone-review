# PhoBERT ABSA + LIME - Phân tích cảm xúc phản hồi điện thoại tiếng Việt

## 1. Giới thiệu

Dự án xây dựng hệ thống phân tích cảm xúc theo khía cạnh đối với bình luận khách hàng về sản phẩm điện thoại tiếng Việt.

Mô hình chính sử dụng **PhoBERT** để dự đoán cảm xúc theo từng khía cạnh và **LIME** để giải thích vì sao mô hình đưa ra dự đoán.

Các khía cạnh phân tích gồm: pin, camera, màn hình, hiệu năng, thiết kế, giá cả, dịch vụ/phụ kiện và đánh giá chung.

---

## 2. Cấu trúc thư mục

```text
phobert-absa-lime-phone-review/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   ├── processed/
│   │   ├── train_absa.csv
│   │   ├── dev_absa.csv
│   │   └── test_absa.csv
│   └── sample/
│       └── sample_upload.csv
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── lime_explain.py
│   ├── baseline.py
│   └── view_data.py + eda_asba.py 
│
├── app/
│   └── app.py
│
├── models/
│   ├── phobert_absa/
│   └── baselines/
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── predictions/
│
├── requirements.txt
├── README.md
└── .gitignore
````

---

## 3. Dữ liệu

Dữ liệu gốc nằm trong:

```text
data/raw/
```

Gồm 3 file:

```text
train.csv
dev.csv
test.csv
```

Các cột chính:

| Cột         | Ý nghĩa                             |
| ----------- | ----------------------------------- |
| `index`     | Số thứ tự                           |
| `comment`   | Bình luận khách hàng                |
| `n_star`    | Số sao đánh giá                     |
| `date_time` | Thời gian bình luận                 |
| `label`     | Nhãn ABSA dạng `{ASPECT#Sentiment}` |

Ví dụ:

```text
{BATTERY#Positive};{CAMERA#Negative};
```

---

## 4. Xem dữ liệu

Xem 10 dòng đầu của file train:

```bash
python src/view_data.py train.csv 10
```

Hoặc dùng pandas:

```python
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
print(df.head())
print(df.info())
```

---

## 5. Tiền xử lý dữ liệu

Chạy:

```bash
python src/preprocess.py
```

File này sẽ chuyển dữ liệu từ dạng gốc sang dạng ABSA:

```text
text, aspect, sentiment, label
```

Kết quả được lưu tại:

```text
data/processed/train_absa.csv
data/processed/dev_absa.csv
data/processed/test_absa.csv
```

Ví dụ sau xử lý:

| text                        | aspect  | sentiment | label |
| --------------------------- | ------- | --------- | ----- |
| Pin tốt nhưng camera hơi mờ | BATTERY | positive  | 2     |
| Pin tốt nhưng camera hơi mờ | CAMERA  | negative  | 0     |

---

## 6. Fine-tune PhoBERT

Chạy:

```bash
python src/train.py
```

File này thực hiện:

* Load dữ liệu đã xử lý
* Fine-tune PhoBERT
* Validation trên tập dev
* Đánh giá trên tập test
* Lưu model vào `models/phobert_absa/`

Kết quả sinh ra:

```text
models/phobert_absa/
outputs/reports/metrics.json
outputs/reports/classification_report.csv
outputs/figures/confusion_matrix.png
outputs/predictions/test_predictions.csv
```

---

## 7. Đánh giá model

Nếu đã train xong và muốn đánh giá lại:

```bash
python src/evaluate.py
```

Các chỉ số gồm:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Weighted F1
* Confusion Matrix

---

## 8. Chạy baseline TF-IDF

Dự án có 2 mô hình baseline để so sánh:

* TF-IDF + Logistic Regression
* TF-IDF + Linear SVM

Chạy:

```bash
python src/baseline_tfidf.py
```

Kết quả lưu tại:

```text
outputs/reports/baseline_comparison.csv
```

---

## 9. Dự đoán cảm xúc

Chạy thử dự đoán:

```bash
python src/predict.py
```

File này load model đã train từ:

```text
models/phobert_absa/
```

và dự đoán cảm xúc cho một bình luận theo từng khía cạnh.

---

## 10. Giải thích bằng LIME

Chạy:

```bash
python src/lime_explain.py
```

LIME dùng để giải thích các từ/cụm từ nào ảnh hưởng đến kết quả dự đoán của PhoBERT.

Lưu ý: LIME không phải mô hình phân loại mới, nên không có Accuracy/F1 riêng.

---

## 11. Chạy demo Streamlit

Chạy:

```bash
streamlit run app/app.py
```

Demo hỗ trợ:

* Nhập một bình luận và chọn khía cạnh
* Dự đoán cảm xúc
* Hiển thị giải thích LIME
* Upload file CSV/XLSX để dự đoán hàng loạt
* Tải kết quả dự đoán

Trước khi chạy demo, cần có model trong:

```text
models/phobert_absa/
```

Nếu chưa có model, cần chạy:

```bash
python src/train.py
```

---

## 12. Cài thư viện

Cài bằng file requirements:

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install pandas numpy scikit-learn matplotlib tqdm openpyxl
pip install torch transformers accelerate
pip install lime streamlit
```

---

## 13. Thứ tự chạy project

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/baseline_tfidf.py
python src/predict.py
python src/lime_explain.py
streamlit run app/app.py
```

---

## 14. Lưu ý

* File dữ liệu gốc đặt trong `data/raw/`.
* File đã xử lý đặt trong `data/processed/`.
* Model PhoBERT sau khi train đặt trong `models/phobert_absa/`.
* Không cần train lại khi chạy `predict.py`, `lime_explain.py` hoặc demo Streamlit.
* Nếu model quá nặng, không nên push lên GitHub mà nên lưu bằng Google Drive.

``````
