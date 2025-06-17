import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from backend.ml.dataset import SyntheticMathDataset
from backend.ml.model import MathClassifier
import time
import os


def train():
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Usando dispositivo: {device} ===\n")

    # Hiperparámetros
    batch_size = 4
    epochs = 3
    learning_rate = 1e-5

    # Dataset y DataLoader
    dataset_path = "backend/ml/data/synthetic_math_dataset.json"
    if not os.path.exists(dataset_path):
        print("❌ Dataset no encontrado.")
        return

    train_dataset = SyntheticMathDataset(file_path=dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Modelo
    model = MathClassifier(num_labels=3)
    model.to(device)

    # Criterio y optimizador
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_time = time.time() - start_time
        accuracy = correct / len(train_dataset)

        print(f"\nÉpoca {epoch+1}/{epochs} | "
              f"Pérdida: {total_loss/len(train_loader):.4f} | "
              f"Precisión: {accuracy:.2%} | "
              f"Tiempo: {epoch_time:.2f}s\n")

    torch.save(model.state_dict(), "backend/ml/model_weights.pt")
    print("✅ Modelo entrenado y guardado en 'backend/ml/model_weights.pt'")


if __name__ == "__main__":
    train()
