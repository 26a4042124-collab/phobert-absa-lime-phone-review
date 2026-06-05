import torch
from torch.utils.data import Dataset


class ABSADataset(Dataset):
    """
    Dataset cho bài toán ABSA.

    Mỗi mẫu gồm:
    - text: bình luận khách hàng
    - aspect: khía cạnh cần phân tích
    - label: sentiment đã mã hóa số

    Input đưa vào tokenizer:
    text, aspect

    Tokenizer sẽ tự tạo dạng phân tách tương ứng cho PhoBERT/RoBERTa.
    """

    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        text = str(row["text"])
        aspect = str(row["aspect"])
        label = int(row["label"])

        encoding = self.tokenizer(
            text,
            aspect,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }