import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def ensure_dir(path):
    """
    Tạo thư mục nếu chưa tồn tại.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(file_path):
    """
    Tạo thư mục cha của một file nếu chưa tồn tại.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def read_csv_safely(file_path):
    """
    Đọc file CSV với nhiều encoding khác nhau để hạn chế lỗi tiếng Việt.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    encodings = ["utf-8-sig", "utf-8", "cp1258", "latin1"]

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Không đọc được file CSV: {file_path}")


def save_csv(df, output_path):
    """
    Lưu DataFrame ra file CSV với encoding utf-8-sig để mở Excel không lỗi tiếng Việt.
    """
    output_path = ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu CSV: {output_path}")


def save_json(data, output_path):
    """
    Lưu dictionary/list ra file JSON.
    """
    output_path = ensure_parent_dir(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu JSON: {output_path}")


def load_json(input_path):
    """
    Đọc file JSON.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    """
    Làm sạch văn bản ở mức vừa phải.
    Không xóa quá mạnh để tránh mất thông tin cảm xúc.
    """
    text = str(text).strip()

    # Xóa URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_sentiment(sentiment):
    """
    Chuẩn hóa nhãn cảm xúc về chữ thường.
    Ví dụ: Positive -> positive
    """
    return str(sentiment).strip().lower()


def parse_absa_labels(label_text, valid_labels=None):
    """
    Tách nhãn ABSA dạng:
    {GENERAL#Positive};{BATTERY#Negative};

    Thành:
    [("GENERAL", "positive"), ("BATTERY", "negative")]
    """
    label_text = str(label_text)

    matches = re.findall(r"\{([^#\}]+)#([^#\}]+)\}", label_text)

    parsed_labels = []

    for aspect, sentiment in matches:
        aspect = str(aspect).strip()
        sentiment = normalize_sentiment(sentiment)

        if valid_labels is None or sentiment in valid_labels:
            parsed_labels.append((aspect, sentiment))

    return parsed_labels


def check_required_columns(df, required_columns, file_name="DataFrame"):
    """
    Kiểm tra DataFrame có đủ cột bắt buộc không.
    """
    missing_columns = []

    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        raise ValueError(
            f"{file_name} thiếu các cột bắt buộc: {missing_columns}. "
            f"Các cột hiện có: {list(df.columns)}"
        )


def print_dataset_info(name, df):
    """
    In thông tin cơ bản của dataset.
    """
    print("=" * 80)
    print(f"Thông tin tập dữ liệu: {name}")
    print(f"Số dòng: {len(df)}")
    print(f"Số cột: {len(df.columns)}")
    print(f"Các cột: {list(df.columns)}")

    if "label" in df.columns:
        print("\nPhân bố label:")
        print(df["label"].value_counts())

    if "sentiment" in df.columns:
        print("\nPhân bố sentiment:")
        print(df["sentiment"].value_counts())

    if "aspect" in df.columns:
        print("\nPhân bố aspect:")
        print(df["aspect"].value_counts())

    print("=" * 80)


def save_confusion_matrix_figure(y_true, y_pred, labels, display_labels, output_path, title):
    """
    Vẽ và lưu confusion matrix.
    """
    output_path = ensure_parent_dir(output_path)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels
    )

    disp.plot(values_format="d")
    plt.title(title)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu confusion matrix: {output_path}")


def format_metrics_percent(metrics):
    """
    Chuyển metrics dạng 0.7873 thành dạng phần trăm 78.73.
    """
    formatted = {}

    for key, value in metrics.items():
        try:
            formatted[key] = round(float(value) * 100, 2)
        except (ValueError, TypeError):
            formatted[key] = value

    return formatted


def load_metrics_as_dataframe(metrics_path):
    """
    Đọc file metrics.json và chuyển thành DataFrame để dễ hiển thị hoặc đưa vào báo cáo.
    """
    metrics = load_json(metrics_path)
    metrics_percent = format_metrics_percent(metrics)

    df = pd.DataFrame([
        {
            "metric": key,
            "value": value
        }
        for key, value in metrics_percent.items()
    ])

    return df


def build_absa_input_text(df):
    """
    Ghép text và aspect thành chuỗi input cho các mô hình baseline TF-IDF.

    Ví dụ:
    Pin tốt nhưng camera mờ [ASP] CAMERA
    """
    check_required_columns(df, ["text", "aspect"], "ABSA DataFrame")

    return (
        df["text"].astype(str)
        + " [ASP] "
        + df["aspect"].astype(str)
    ).tolist()


def get_project_root():
    """
    Trả về thư mục gốc project.
    File utils.py nằm trong src/, nên parent.parent là project root.
    """
    return Path(__file__).resolve().parent.parent