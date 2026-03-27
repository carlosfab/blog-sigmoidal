"""
Métricas de Avaliação em Modelos de Classificação
Acurácia, Precisão, Recall, F1-Score e AUC-ROC na prática com Python
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
import json
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. Carregar o dataset Breast Cancer Wisconsin
# ============================================================
print("[INFO] Carregando dataset Breast Cancer Wisconsin...")
dados = load_breast_cancer()
X, y = dados.data, dados.target
nomes_classes = dados.target_names  # ['malignant', 'benign']

print(f"[INFO] Amostras: {X.shape[0]}")
print(f"[INFO] Features: {X.shape[1]}")
print(f"[INFO] Classes: {nomes_classes}")
print(f"[INFO] Distribuição: Maligno={sum(y==0)}, Benigno={sum(y==1)}")

# ============================================================
# 2. Preparar os dados
# ============================================================
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)

print(f"[INFO] Treino: {X_treino.shape[0]} amostras")
print(f"[INFO] Teste:  {X_teste.shape[0]} amostras")

# ============================================================
# 3. Treinar um modelo de Regressão Logística
# ============================================================
print("\n[INFO] Treinando Regressão Logística...")
modelo = LogisticRegression(random_state=42, max_iter=1000)
modelo.fit(X_treino, y_treino)

y_pred = modelo.predict(X_teste)
y_proba = modelo.predict_proba(X_teste)[:, 1]

# ============================================================
# 4. Calcular todas as métricas
# ============================================================
acc = accuracy_score(y_teste, y_pred)
prec = precision_score(y_teste, y_pred)
rec = recall_score(y_teste, y_pred)
f1 = f1_score(y_teste, y_pred)
auc = roc_auc_score(y_teste, y_proba)

print(f"\n{'='*40}")
print(f"  Acurácia:  {acc:.4f}")
print(f"  Precisão:  {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc:.4f}")
print(f"{'='*40}")

# ============================================================
# 5. Relatório de classificação completo
# ============================================================
report = classification_report(
    y_teste, y_pred, target_names=["Maligno", "Benigno"]
)
print(f"\n[INFO] Relatório de Classificação:\n{report}")

# ============================================================
# 6. Matriz de confusão
# ============================================================
cm = confusion_matrix(y_teste, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Maligno", "Benigno"],
            yticklabels=["Maligno", "Benigno"],
            annot_kws={"size": 16})
ax.set_xlabel("Predição", fontsize=13)
ax.set_ylabel("Valor Real", fontsize=13)
ax.set_title("Matriz de Confusão", fontsize=14, fontweight="bold")

# Anotar TP, TN, FP, FN
tp = cm[1, 1]
tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]
ax.text(0.5, -0.15, f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        transform=ax.transAxes, ha="center", fontsize=11, color="gray")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "metricas-matriz-confusao.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: metricas-matriz-confusao.png")

# ============================================================
# 7. Curva ROC
# ============================================================
fpr, tpr, thresholds = roc_curve(y_teste, y_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2.5, color="#2563eb",
        label=f"Regressão Logística (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
        label="Classificador aleatório (AUC = 0.500)")
ax.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
ax.set_title("Curva ROC", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "metricas-curva-roc.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: metricas-curva-roc.png")

# ============================================================
# 8. Comparação visual das métricas
# ============================================================
metricas_nomes = ["Acurácia", "Precisão", "Recall", "F1-Score", "AUC-ROC"]
metricas_valores = [acc, prec, rec, f1, auc]

fig, ax = plt.subplots(figsize=(10, 5))
cores = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
bars = ax.bar(metricas_nomes, metricas_valores, color=cores, width=0.6, edgecolor="white")

for bar, val in zip(bars, metricas_valores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=13, fontweight="bold")

ax.set_ylim(0, 1.12)
ax.set_ylabel("Valor da Métrica", fontsize=12)
ax.set_title("Comparação das Métricas de Classificação", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "metricas-comparacao.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: metricas-comparacao.png")

# ============================================================
# 9. Impacto do threshold na Precisão e Recall
# ============================================================
thresholds_range = np.linspace(0.01, 0.99, 100)
precisions = []
recalls = []
f1s = []

for t in thresholds_range:
    y_pred_t = (y_proba >= t).astype(int)
    if y_pred_t.sum() == 0:
        precisions.append(1.0)
    else:
        precisions.append(precision_score(y_teste, y_pred_t, zero_division=1))
    recalls.append(recall_score(y_teste, y_pred_t, zero_division=0))
    f1s.append(f1_score(y_teste, y_pred_t, zero_division=0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds_range, precisions, label="Precisão", linewidth=2, color="#10b981")
ax.plot(thresholds_range, recalls, label="Recall", linewidth=2, color="#f59e0b")
ax.plot(thresholds_range, f1s, label="F1-Score", linewidth=2, color="#ef4444",
        linestyle="--")
ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="Threshold = 0.5")
ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("Valor da Métrica", fontsize=12)
ax.set_title("Precisão vs. Recall em Função do Threshold", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "metricas-threshold.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: metricas-threshold.png")

# Salvar métricas para o artigo
metricas_dict = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc_roc": auc,
    "n_amostras": X.shape[0],
    "n_features": X.shape[1],
    "n_treino": X_treino.shape[0],
    "n_teste": X_teste.shape[0],
    "cm": cm.tolist(),
}
with open(os.path.join(SAVE_DIR, "metricas.json"), "w") as f:
    json.dump(metricas_dict, f, indent=2)

print("\n[INFO] Concluído!")
