import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    PREDICTION_DIR,
    MAX_LENGTH,
    BATCH_SIZE,
    ID2LABEL
)

from dataset import ABSADataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_directories():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data():
    test_path = PROCESSED_DATA_DIR / "test_absa.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file {test_path}. "
            f"Hãy chạy trước: python src/preprocess.py"
        )

    df = pd.read_csv(test_path, encoding="utf-8-sig")

    required_columns = ["text", "aspect", "sentiment", "label"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"File test_absa.csv thiếu cột: {col}")

    df["label"] = df["label"].astype(int)

    print(f"Số dòng test: {len(df)}")

    return df


def load_model_and_tokenizer():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại {MODEL_DIR}. "
            f"Hãy train trước bằng: python src/train.py"
        )

    print(f"Đang load model từ: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)
    model.eval()

    print(f"Model đã load xong. Device: {device}")

    return model, tokenizer


def predict_test_set(model, dataloader):
    all_true_labels = []
    all_pred_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    return (
        np.array(all_true_labels),
        np.array(all_pred_labels),
        np.array(all_probabilities)
    )


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

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1)
    }

    return metrics


def save_metrics(metrics):
    metrics_path = REPORT_DIR / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu metrics tại: {metrics_path}")


def save_classification_report(y_true, y_pred):
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

    report_path = REPORT_DIR / "classification_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")

    print(f"Đã lưu classification report tại: {report_path}")


def save_confusion_matrix(y_true, y_pred):
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
    plt.title("Confusion Matrix - PhoBERT ABSA")

    cm_path = FIGURE_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu confusion matrix tại: {cm_path}")


def save_predictions(test_df, y_true, y_pred, probabilities):
    result_df = test_df.copy()

    result_df["true_label"] = [ID2LABEL[int(x)] for x in y_true]
    result_df["pred_label"] = [ID2LABEL[int(x)] for x in y_pred]

    result_df["prob_negative"] = probabilities[:, 0]
    result_df["prob_neutral"] = probabilities[:, 1]
    result_df["prob_positive"] = probabilities[:, 2]

    result_df["confidence"] = probabilities.max(axis=1)

    predictions_path = PREDICTION_DIR / "test_predictions.csv"
    result_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print(f"Đã lưu dự đoán test tại: {predictions_path}")


def evaluate():
    ensure_directories()

    print("1. Đang tải dữ liệu test...")
    test_df = load_test_data()

    print("2. Đang tải model và tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print("3. Đang tạo dataset và dataloader...")
    test_dataset = ABSADataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("4. Đang dự đoán trên tập test...")
    y_true, y_pred, probabilities = predict_test_set(
        model=model,
        dataloader=test_loader
    )

    print("5. Đang tính metrics...")
    metrics = compute_metrics(y_true, y_pred)

    print("\nKẾT QUẢ ĐÁNH GIÁ:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n6. Đang lưu kết quả...")
    save_metrics(metrics)
    save_classification_report(y_true, y_pred)
    save_confusion_matrix(y_true, y_pred)
    save_predictions(test_df, y_true, y_pred, probabilities)

    print("\nHoàn tất evaluate model.")


if __name__ == "__main__":
    evaluate()