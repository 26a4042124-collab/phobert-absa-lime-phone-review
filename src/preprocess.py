import re
from pathlib import Path

import pandas as pd

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LABEL2ID,
    DEBUG,
    DEBUG_SAMPLE_SIZE
)


def ensure_directories():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """
    Đọc CSV với nhiều encoding phổ biến để tránh lỗi tiếng Việt.
    """
    encodings = ["utf-8-sig", "utf-8", "cp1258", "latin1"]

    for encoding in encodings:
        try:
            print(f"Đang đọc {file_path} với encoding={encoding}")
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "Không đọc được file CSV với các encoding phổ biến.",
        b"",
        0,
        1,
        "encoding error"
    )


def clean_text(text: str) -> str:
    """
    Làm sạch bình luận ở mức vừa phải.
    Không xóa quá mạnh để tránh mất thông tin cảm xúc.
    """
    text = str(text).strip()

    # Xóa URL nếu có
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_sentiment(sentiment: str) -> str:
    """
    Chuẩn hóa sentiment về lowercase.
    Ví dụ: Positive -> positive
    """
    return str(sentiment).strip().lower()


def parse_absa_labels(label_text: str):
    """
    Tách nhãn dạng:
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

        if sentiment in LABEL2ID:
            parsed_labels.append((aspect, sentiment))

    return parsed_labels


def convert_raw_file_to_absa(input_path: Path, output_path: Path, sample_size=None) -> pd.DataFrame:
    """
    Convert một file raw train/dev/test sang format:
    text, aspect, sentiment, label
    """
    print("=" * 80)
    print(f"Đang xử lý file: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    df_raw = read_csv_safely(input_path)

    required_columns = ["comment", "label"]

    for col in required_columns:
        if col not in df_raw.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {col}. Các cột hiện có: {list(df_raw.columns)}")

    df_raw = df_raw.dropna(subset=["comment", "label"])

    if sample_size is not None and len(df_raw) > sample_size:
        df_raw = df_raw.sample(sample_size, random_state=42)

    parsed_data = []

    for _, row in df_raw.iterrows():
        comment = clean_text(row["comment"])
        label_str = row["label"]

        aspect_sentiment_pairs = parse_absa_labels(label_str)

        for aspect, sentiment in aspect_sentiment_pairs:
            parsed_data.append({
                "text": comment,
                "aspect": aspect,
                "sentiment": sentiment,
                "label": LABEL2ID[sentiment]
            })

    df_processed = pd.DataFrame(parsed_data)

    if df_processed.empty:
        raise ValueError(f"Không parse được dòng ABSA nào từ file: {input_path}")

    df_processed = df_processed.dropna(subset=["text", "aspect", "sentiment", "label"])
    df_processed["label"] = df_processed["label"].astype(int)

    df_processed.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Đã lưu file xử lý: {output_path}")
    print(f"Số dòng raw ban đầu: {len(df_raw)}")
    print(f"Số dòng sau convert ABSA: {len(df_processed)}")
    print("Ví dụ dòng đầu tiên:")
    print(df_processed.iloc[0].to_dict())

    return df_processed


def main():
    ensure_directories()

    sample_size = DEBUG_SAMPLE_SIZE if DEBUG else None

    files = {
        "train.csv": "train_absa.csv",
        "dev.csv": "dev_absa.csv",
        "test.csv": "test_absa.csv"
    }

    for raw_name, processed_name in files.items():
        input_path = RAW_DATA_DIR / raw_name
        output_path = PROCESSED_DATA_DIR / processed_name

        # Chỉ sample train khi DEBUG, không sample dev/test
        current_sample_size = sample_size if raw_name == "train.csv" else None

        convert_raw_file_to_absa(
            input_path=input_path,
            output_path=output_path,
            sample_size=current_sample_size
        )

    print("=" * 80)
    print("Hoàn tất tiền xử lý và convert dữ liệu ABSA.")


if __name__ == "__main__":
    main()