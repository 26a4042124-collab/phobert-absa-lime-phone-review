import numpy as np
import torch
from pathlib import Path
from lime.lime_text import LimeTextExplainer

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MODEL_DIR, ID2LABEL, LABEL2ID, MAX_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ABSALimeExplainer:
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

        self.class_names = [
            ID2LABEL[0],
            ID2LABEL[1],
            ID2LABEL[2]
        ]

        self.explainer = LimeTextExplainer(
            class_names=self.class_names
        )

        print(f"Model và LIME đã load xong. Device: {device}")

    def predict_proba_for_lime(self, texts, fixed_aspect):
        """
        Hàm bắt buộc cho LIME.

        LIME sẽ tạo nhiều biến thể của comment.
        Với mỗi biến thể, ta ghép aspect cố định vào tokenizer.
        """

        all_probs = []

        for text in texts:
            encoding = self.tokenizer(
                str(text),
                str(fixed_aspect),
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

                probs = torch.softmax(outputs.logits, dim=1)

            all_probs.append(probs.cpu().numpy()[0])

        return np.array(all_probs)

    def predict_one(self, text, aspect):
        """
        Dự đoán sentiment trước khi giải thích.
        """

        probs = self.predict_proba_for_lime([text], aspect)[0]
        pred_id = int(np.argmax(probs))

        return {
            "aspect": aspect,
            "pred_label": ID2LABEL[pred_id],
            "confidence": float(probs[pred_id]),
            "prob_negative": float(probs[0]),
            "prob_neutral": float(probs[1]),
            "prob_positive": float(probs[2])
        }

    def explain(self, text, aspect, num_features=10, num_samples=300):
        """
        Giải thích dự đoán của model bằng LIME.

        text: bình luận gốc
        aspect: khía cạnh cần giải thích
        """

        prediction = self.predict_one(text, aspect)
        pred_label = prediction["pred_label"]
        pred_id = LABEL2ID[pred_label]

        explanation = self.explainer.explain_instance(
            text_instance=str(text),
            classifier_fn=lambda texts: self.predict_proba_for_lime(
                texts,
                fixed_aspect=aspect
            ),
            labels=[pred_id],
            num_features=num_features,
            num_samples=num_samples
        )

        lime_result = explanation.as_list(label=pred_id)

        result = {
            "text": text,
            "aspect": aspect,
            "pred_label": pred_label,
            "confidence": prediction["confidence"],
            "lime_words": lime_result
        }

        return result

# =================================================================
# === HÀM NÀY ĐỂ KẾT NỐI VỚI STREAMLIT APP.PY ===
_global_explainer = None

def get_lime_explanation(text, aspect):
    global _global_explainer
    # Load model 1 lần duy nhất để tránh treo máy
    if _global_explainer is None:
        _global_explainer = ABSALimeExplainer()
        
    # Tính toán dự đoán
    prediction = _global_explainer.predict_one(text, aspect)
    pred_label = prediction["pred_label"]
    pred_id = LABEL2ID.get(pred_label, 0) # Lấy ID của nhãn

    # Chạy LIME
    explanation = _global_explainer.explainer.explain_instance(
        text_instance=str(text),
        classifier_fn=lambda texts: _global_explainer.predict_proba_for_lime(texts, fixed_aspect=aspect),
        labels=[pred_id],
        num_features=10,
        num_samples=300
    )
    
    # Ép LIME xuất ra định dạng HTML để Streamlit hiển thị được
    return explanation.as_html()
# =================================================================
if __name__ == "__main__":
    explainer = ABSALimeExplainer()

    comment = "Pin dùng ổn nhưng camera chụp đêm hơi tệ, giao hàng nhanh."
    aspect = "CAMERA"

    result = explainer.explain(
        text=comment,
        aspect=aspect,
        num_features=10,
        num_samples=300
    )

    print("\nKẾT QUẢ DỰ ĐOÁN")
    print("Bình luận:", result["text"])
    print("Aspect:", result["aspect"])
    print("Dự đoán:", result["pred_label"])
    print("Độ tin cậy:", round(result["confidence"], 4))

    print("\nTỪ/CỤM TỪ ẢNH HƯỞNG THEO LIME")
    for word, weight in result["lime_words"]:
        print(f"{word}: {weight:.4f}")

if __name__ == "__main__":
    explainer = ABSALimeExplainer()

    comment = "Pin dùng ổn nhưng camera chụp đêm hơi tệ, giao hàng nhanh."
    aspect = "BATTERY"

    result = explainer.explain(
        text=comment,
        aspect=aspect,
        num_features=10,
        num_samples=300
    )

    print("\nKẾT QUẢ DỰ ĐOÁN:")
    print("Bình luận:", result["text"])
    print("Aspect:", result["aspect"])
    print("Dự đoán:", result["pred_label"])
    print("Độ tin cậy:", round(result["confidence"], 4))

    print("\nTỪ/CỤM TỪ ẢNH HƯỞNG THEO LIME:")
    for word, weight in result["lime_words"]:
        print(f"{word}: {weight:.4f}")