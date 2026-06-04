import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import (
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    ID2LABEL
)


BASELINE_MODEL_DIR = Path("models") / "baselines"


def ensure_directories():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train_path = PROCESSED_DATA_DIR / "train_absa.csv"
    dev_path = PROCESSED_DATA_DIR / "dev_absa.csv"
    test_path = PROCESSED_DATA_DIR / "test_absa.csv"

    for path in [train_path, dev_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {path}. "
                f"Hãy chạy trước: python src/preprocess.py"
            )

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    dev_df = pd.read_csv(dev_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")

    required_columns = ["text", "aspect", "label"]

    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"File {name}_absa.csv thiếu cột: {col}")

    train_df["label"] = train_df["label"].astype(int)
    dev_df["label"] = dev_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    print("Số dòng dữ liệu:")
    print(f"Train: {len(train_df)}")
    print(f"Dev:   {len(dev_df)}")
    print(f"Test:  {len(test_df)}")

    return train_df, dev_df, test_df


def build_input_text(df):
    """
    Ghép bình luận và aspect thành một chuỗi đầu vào cho TF-IDF.
    Mô hình baseline không hiểu sentence pair như PhoBERT,
    nên ta ghép thủ công: text + [ASP] + aspect.
    """
    return (df["text"].astype(str) + " [ASP] " + df["aspect"].astype(str)).tolist()


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1)
    }


def save_classification_report(model_name, y_true, y_pred):
    target_names = [
        ID2LABEL[0],
        ID2LABEL[1],
        ID2LABEL[2]
    ]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = REPORT_DIR / f"{model_name}_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")

    print(f"Đã lưu classification report: {report_path}")


def save_confusion_matrix(model_name, y_true, y_pred):
    labels = [0, 1, 2]
    display_labels = [
        ID2LABEL[0],
        ID2LABEL[1],
        ID2LABEL[2]
    ]

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
    plt.title(f"Confusion Matrix - {model_name}")

    cm_path = FIGURE_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu confusion matrix: {cm_path}")


def train_and_evaluate_model(model_name, pipeline, x_train, y_train, x_test, y_test):
    print("=" * 80)
    print(f"Đang train mô hình: {model_name}")

    pipeline.fit(x_train, y_train)

    print(f"Đang đánh giá mô hình: {model_name}")
    y_pred = pipeline.predict(x_test)

    metrics = compute_metrics(y_test, y_pred)

    print(f"\nKẾT QUẢ {model_name}:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    save_classification_report(model_name, y_test, y_pred)
    save_confusion_matrix(model_name, y_test, y_pred)

    model_path = BASELINE_MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    print(f"Đã lưu model baseline: {model_path}")

    return metrics


def main():
    ensure_directories()

    train_df, dev_df, test_df = load_data()

    # Dùng train để học, test để đánh giá cuối.
    # Dev có thể dùng để thử tham số, nhưng bản baseline đơn giản này chưa tune sâu.
    x_train = build_input_text(train_df)
    y_train = train_df["label"].tolist()

    x_test = build_input_text(test_df)
    y_test = test_df["label"].tolist()

    models = {
        "tfidf_logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=50000
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1
            ))
        ]),

        "tfidf_linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=50000
            )),
            ("clf", LinearSVC(
                class_weight="balanced",
                max_iter=5000
            ))
        ])
    }

    comparison_rows = []

    for model_name, pipeline in models.items():
        metrics = train_and_evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

        row = {
            "model": model_name,
            **metrics
        }

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_path = REPORT_DIR / "baseline_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("BẢNG SO SÁNH BASELINE:")
    print(comparison_df)

    print(f"\nĐã lưu bảng so sánh tại: {comparison_path}")
    print("Hoàn tất train và evaluate 2 mô hình TF-IDF baseline.")


if __name__ == "__main__":
    main()