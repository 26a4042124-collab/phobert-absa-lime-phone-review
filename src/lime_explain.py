import torch
import torch.nn.functional as F
import numpy as np
from lime.lime_text import LimeTextExplainer
import sys
import os

# Ép Python hiểu thư mục hiện tại để import không bị lỗi
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# CHỈ import model và tokenizer, KHÔNG import device nữa để tránh lỗi
try:
    from predict import model, tokenizer
except ImportError:
    from src.predict import model, tokenizer

# Tự khởi tạo device riêng trực tiếp tại đây!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

explainer = LimeTextExplainer(class_names=['Tiêu cực', 'Trung tính', 'Tích cực'])

def predict_proba_for_lime(texts):
    if model is None or tokenizer is None:
        return np.zeros((len(texts), 3))
        
    all_probs = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", max_length=256, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
        all_probs.append(probs.cpu().numpy())
        
    return np.vstack(all_probs)

def get_lime_explanation(comment, aspect):
    if model is None:
        return "<div style='padding:20px; background:#ffebee; color:#c62828; border-radius:8px;'><b>Chưa tải được mô hình!</b><br>Vui lòng chạy file huấn luyện (03_train_test.ipynb) để tạo mô hình trước khi dùng LIME.</div>"
        
    try:
        text_to_explain = f"{comment} </s></s> {aspect}"
        exp = explainer.explain_instance(
            text_instance=text_to_explain,
            classifier_fn=predict_proba_for_lime,
            num_features=10,
            num_samples=100
        )
        return exp.as_html()
    except Exception as e:
        print(f"Lỗi khi chạy LIME: {e}")
        return "<div style='color:red;'>Lỗi trong quá trình tạo giải thích LIME.</div>"