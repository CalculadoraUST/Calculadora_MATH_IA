import json
from torch.utils.data import Dataset
from transformers import BertTokenizer

class SyntheticMathDataset(Dataset):
    def __init__(self, file_path="backend/ml/data/synthetic_math_dataset.json", tokenizer_name='bert-base-uncased', max_length=128):
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            for doc in raw:
                for page in doc["pages"]:
                    for item in page["content"]:
                        if item["type"] == "equation":
                            # Puedes agregar aquí lógica para asignar etiquetas si las tienes
                            self.data.append({
                                "equation": item["value"],
                                "format": item.get("format", ""),
                                # "label": ... # Si tienes etiquetas, agrégalas aquí
                            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        equation = self.data[idx]["equation"]
        # Si tienes etiquetas, descomenta la siguiente línea
        # label = self.data[idx]["label"]
        encoding = self.tokenizer(
            equation,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        # Si tienes etiquetas, retorna también label
        # return {**{k: v.squeeze(0) for k, v in encoding.items()}, "label": label}
        return {k: v.squeeze(0) for k, v in encoding.items()}