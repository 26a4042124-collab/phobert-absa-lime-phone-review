import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. KHAI BÁO CÁC BIẾN TOÀN CỤC Ở ĐÂY ĐỂ TRÁNH LỖI IMPORT CHO CÁC FILE KHÁC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
tokenizer = None

MODEL_PATH = "models/phobert_absa"
MODEL_NAME = "vinai/phobert-base"

try:
    print("Đang tải Tokenizer và Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    # Số lượng class: 3 (Tích cực, Tiêu cực, Trung tính)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
    model.to(device)
    model.eval() # Chuyển sang chế độ dự đoán
    print("Tải Model thành công!")
except Exception as e:
    print(f"Lưu ý: Chưa tải được model thật (Có thể bạn chưa Train). Chi tiết: {e}")

# Định nghĩa map nhãn
ID_TO_LABEL = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}

def predict_sentiment(comment, aspect):
    if model is None or tokenizer is None:
        return "Chưa có Model", 0.0
        
    try:
        text = f"{comment} </s></s> {aspect}"
        inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        probs = F.softmax(logits, dim=1)
        predicted_class_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class_id].item() * 100
        
        predicted_label = ID_TO_LABEL[predicted_class_id]
        return predicted_label, confidence
        
    except Exception as e:
        print(f"Lỗi dự đoán: {e}")
        return "Lỗi", 0.0