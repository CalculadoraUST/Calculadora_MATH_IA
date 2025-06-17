from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json

class SyntheticMathDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = [d for d in data if d.get("input") and d.get("output")]
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.label2id = {"derivada": 0, "integral": 1, "limite": 2}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoded = self.tokenizer(
            item["input"],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": self.label2id[item["output"]]
        }
