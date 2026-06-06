import json
import random
import inspect

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    PREDICTION_DIR,
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LABEL2ID,
    ID2LABEL
)

from dataset import ABSADataset


def ensure_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_data():
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

    required_columns = ["text", "aspect", "sentiment", "label"]

    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"File {name} thiếu cột {col}")

        df["label"] = df["label"].astype(int)

    print("Số dòng dữ liệu:")
    print(f"Train: {len(train_df)}")
    print(f"Dev:   {len(dev_df)}")
    print(f"Test:  {len(test_df)}")

    return train_df, dev_df, test_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }


def build_training_arguments():
    """
    Tương thích với nhiều version transformers.
    Một số bản dùng evaluation_strategy, một số bản mới có thể dùng eval_strategy.
    """
    base_args = {
        "output_dir": str(OUTPUT_DIR / "trainer_results"),
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_dir": str(OUTPUT_DIR / "logs"),
        "logging_steps": 50,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none"
    }

    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters

    if "evaluation_strategy" in parameters:
        base_args["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in parameters:
        base_args["eval_strategy"] = "epoch"

    return TrainingArguments(**base_args)


def save_test_outputs(trainer, test_dataset, test_df):
    print("Đang dự đoán trên tập test...")

    prediction_output = trainer.predict(test_dataset)

    logits = prediction_output.predictions
    true_ids = prediction_output.label_ids
    pred_ids = np.argmax(logits, axis=-1)

    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    max_probs = probabilities.max(axis=1)

    test_result_df = test_df.copy()
    test_result_df["true_label"] = [ID2LABEL[int(x)] for x in true_ids]
    test_result_df["pred_label"] = [ID2LABEL[int(x)] for x in pred_ids]
    test_result_df["confidence"] = max_probs

    prediction_path = PREDICTION_DIR / "test_predictions.csv"
    test_result_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    print(f"Đã lưu dự đoán test tại: {prediction_path}")

    report = classification_report(
        true_ids,
        pred_ids,
        target_names=["negative", "neutral", "positive"],
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = REPORT_DIR / "classification_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")

    print(f"Đã lưu classification report tại: {report_path}")

    metrics = compute_metrics((logits, true_ids))

    metrics_path = REPORT_DIR / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu metrics tại: {metrics_path}")

    cm = confusion_matrix(true_ids, pred_ids)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "neutral", "positive"]
    )

    disp.plot(values_format="d")
    plt.title("Confusion Matrix - PhoBERT ABSA")

    cm_path = FIGURE_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu confusion matrix tại: {cm_path}")

    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # Chuyển dữ liệu qua model
        outputs = model(**inputs)
        logits = outputs.logits
        device = model.device
        class_weights = torch.tensor([1.5, 3.0, 0.5]).to(device) 

        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def train():
    ensure_directories()
    set_seed(42)

    print("1. Đang tải dữ liệu đã xử lý...")
    train_df, dev_df, test_df = load_processed_data()
    
    print("2. Đang tải PhoBERT tokenizer và model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        use_safetensors=False
    )

    print("3. Đang tạo Dataset...")
    train_dataset = ABSADataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    dev_dataset = ABSADataset(
        dataframe=dev_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    test_dataset = ABSADataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    print("4. Đang cấu hình Trainer...")
    training_args = build_training_arguments()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )

    print("5. BẮT ĐẦU FINE-TUNE PHOBERT...")
    trainer.train()

    print("6. Lưu model tốt nhất...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print(f"Model đã lưu tại: {MODEL_DIR}")

    print("7. Đánh giá model trên tập test...")
    save_test_outputs(
        trainer=trainer,
        test_dataset=test_dataset,
        test_df=test_df
    )

    print("Hoàn tất fine-tune PhoBERT.")


if __name__ == "__main__":
    train()