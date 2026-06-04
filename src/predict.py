import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MODEL_DIR, ID2LABEL, MAX_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ABSAPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model tại {self.model_dir}. "
                f"Hãy train model trước hoặc copy thư mục models/phobert_absa vào đúng vị trí."
            )

        print(f"Đang load model từ: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            use_fast=False
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        )

        self.model.to(device)
        self.model.eval()

        print(f"Model đã load xong. Device: {device}")

    def predict(self, text: str, aspect: str):
        """
        Dự đoán sentiment cho một bình luận và một aspect.
        """

        encoding = self.tokenizer(
            str(text),
            str(aspect),
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()

        result = {
            "text": text,
            "aspect": aspect,
            "pred_label": ID2LABEL[pred_id],
            "confidence": float(probs[0][pred_id].cpu().item()),
            "prob_negative": float(probs[0][0].cpu().item()),
            "prob_neutral": float(probs[0][1].cpu().item()),
            "prob_positive": float(probs[0][2].cpu().item())
        }

        return result

    def predict_many_aspects(self, text: str, aspects: list):
        """
        Dự đoán nhiều aspect cho cùng một bình luận.
        """

        results = []

        for aspect in aspects:
            result = self.predict(text, aspect)
            results.append(result)

        return pd.DataFrame(results)


if __name__ == "__main__":
    predictor = ABSAPredictor()

    comment = "Pin dùng ổn nhưng camera chụp đêm hơi tệ, giao hàng nhanh."

    aspects = [
        "GENERAL",
        "BATTERY",
        "CAMERA",
        "SCREEN",
        "PERFORMANCE",
        "DESIGN",
        "PRICE",
        "SER&ACC"
    ]

    result_df = predictor.predict_many_aspects(comment, aspects)

    print("\nKẾT QUẢ DỰ ĐOÁN:")
    print(result_df)

    output_path = "outputs/predictions/sample_prediction.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nĐã lưu kết quả tại: {output_path}")

