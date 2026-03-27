"""
Redes Neurais Multicamadas com PyTorch
Classificação de dígitos manuscritos (MNIST)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. Carregar o dataset MNIST
# ============================================================
print("[INFO] Importando MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
data = mnist.data.astype("float32") / 255.0
labels = mnist.target.astype("int")

print(f"[INFO] Número de imagens: {data.shape[0]}")
print(f"[INFO] Pixels por imagem: {data.shape[1]}")

# ============================================================
# 2. Visualizar exemplos do dataset
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
np.random.seed(42)
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(data))
    ax.imshow(data[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {labels[idx]}", fontsize=12)
    ax.axis("off")
plt.suptitle("Exemplos do Dataset MNIST", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mnist-exemplos.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: mnist-exemplos.png")

# ============================================================
# 3. Dividir em treino (75%) e teste (25%)
# ============================================================
trainX, testX, trainY, testY = train_test_split(
    data, labels, test_size=0.25, random_state=42, stratify=labels
)

print(f"[INFO] Treino: {trainX.shape[0]} imagens")
print(f"[INFO] Teste:  {testX.shape[0]} imagens")

# Converter para tensores PyTorch
trainX_t = torch.tensor(trainX)
testX_t = torch.tensor(testX)
trainY_t = torch.tensor(trainY, dtype=torch.long)
testY_t = torch.tensor(testY, dtype=torch.long)

# DataLoaders
train_dataset = TensorDataset(trainX_t, trainY_t)
test_dataset = TensorDataset(testX_t, testY_t)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ============================================================
# 4. Definir a arquitetura da rede neural
# ============================================================
class RedeNeural(nn.Module):
    """MLP com 2 camadas ocultas: 784 -> 128 -> 64 -> 10"""
    def __init__(self):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.modelo(x)

modelo = RedeNeural()
print(f"\n[INFO] Arquitetura da rede:\n{modelo}")

# Contar parâmetros
total_params = sum(p.numel() for p in modelo.parameters())
print(f"[INFO] Total de parâmetros: {total_params:,}")

# ============================================================
# 5. Treinar o modelo
# ============================================================
criterio = nn.CrossEntropyLoss()
otimizador = optim.SGD(modelo.parameters(), lr=0.01)

EPOCHS = 30
historico = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("\n[INFO] Treinando a rede neural...")
for epoch in range(EPOCHS):
    # --- Treino ---
    modelo.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        otimizador.zero_grad()
        outputs = modelo(X_batch)
        loss = criterio(outputs, y_batch)
        loss.backward()
        otimizador.step()

        running_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # --- Validação ---
    modelo.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = modelo(X_batch)
            loss = criterio(outputs, y_batch)
            val_loss_sum += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    historico["train_loss"].append(train_loss)
    historico["val_loss"].append(val_loss)
    historico["train_acc"].append(train_acc)
    historico["val_acc"].append(val_acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} — "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

print(f"\n[INFO] Acurácia final no teste: {historico['val_acc'][-1]:.2%}")

# ============================================================
# 6. Plotar curvas de treino
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(historico["train_loss"], label="Treino", linewidth=2)
ax1.plot(historico["val_loss"], label="Teste", linewidth=2)
ax1.set_xlabel("Época")
ax1.set_ylabel("Loss")
ax1.set_title("Loss por Época")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(historico["train_acc"], label="Treino", linewidth=2)
ax2.plot(historico["val_acc"], label="Teste", linewidth=2)
ax2.set_xlabel("Época")
ax2.set_ylabel("Acurácia")
ax2.set_title("Acurácia por Época")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Treinamento da Rede Neural — MNIST", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mnist-treinamento.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: mnist-treinamento.png")

# ============================================================
# 7. Relatório de classificação
# ============================================================
modelo.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = modelo(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

report = classification_report(all_labels, all_preds)
print(f"\n[INFO] Relatório de classificação:\n{report}")

# ============================================================
# 8. Matriz de confusão
# ============================================================
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=range(10), yticklabels=range(10))
ax.set_xlabel("Predição", fontsize=12)
ax.set_ylabel("Valor Real", fontsize=12)
ax.set_title("Matriz de Confusão — MNIST", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mnist-matriz-confusao.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: mnist-matriz-confusao.png")

# ============================================================
# 9. Visualizar predições (acertos e erros)
# ============================================================
erros_idx = np.where(all_preds != all_labels)[0]
acertos_idx = np.where(all_preds == all_labels)[0]

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Exemplos de Predições", fontsize=14, fontweight="bold")

# Linha 1: acertos
np.random.seed(42)
for i, ax in enumerate(axes[0]):
    idx = np.random.choice(acertos_idx)
    img = testX[idx].reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"✓ Pred: {all_preds[idx]}", color="green", fontsize=11)
    ax.axis("off")
axes[0][0].set_ylabel("Acertos", fontsize=12)

# Linha 2: erros
for i, ax in enumerate(axes[1]):
    if i < len(erros_idx):
        idx = erros_idx[i]
        img = testX[idx].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"✗ Pred: {all_preds[idx]} (Real: {all_labels[idx]})",
                     color="red", fontsize=10)
    ax.axis("off")
axes[1][0].set_ylabel("Erros", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mnist-predicoes.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: mnist-predicoes.png")

# Salvar métricas para o artigo
metricas = {
    "accuracy_final": historico["val_acc"][-1],
    "train_acc_final": historico["train_acc"][-1],
    "epochs": EPOCHS,
    "total_params": total_params,
    "n_treino": trainX.shape[0],
    "n_teste": testX.shape[0],
}
with open(os.path.join(SAVE_DIR, "metricas.json"), "w") as f:
    json.dump(metricas, f, indent=2)

print("\n[INFO] Concluído!")
