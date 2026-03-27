"""
Distribuições Estatísticas com Python
Normal, Binomial, Poisson, Exponencial e Uniforme na prática
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import json
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Estilo dos gráficos
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ============================================================
# 1. Distribuição Normal (Gaussiana)
# ============================================================
print("[INFO] Gerando Distribuição Normal...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF com diferentes parâmetros
x = np.linspace(-8, 12, 500)
params = [(0, 1, "#2563eb"), (0, 2, "#10b981"), (3, 1.5, "#ef4444")]

for mu, sigma, cor in params:
    y = stats.norm.pdf(x, mu, sigma)
    axes[0].plot(x, y, linewidth=2.5, color=cor,
                 label=f"μ={mu}, σ={sigma}")

axes[0].set_title("Distribuição Normal — PDF", fontsize=13, fontweight="bold")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Densidade de Probabilidade")
axes[0].legend(fontsize=10)

# Histograma de amostras
np.random.seed(42)
amostras = np.random.normal(loc=170, scale=8, size=5000)
axes[1].hist(amostras, bins=50, density=True, alpha=0.7, color="#2563eb",
             edgecolor="white", label="Amostras (n=5000)")
x_fit = np.linspace(140, 200, 200)
axes[1].plot(x_fit, stats.norm.pdf(x_fit, 170, 8), "r-", linewidth=2.5,
             label="PDF teórica (μ=170, σ=8)")
axes[1].set_title("Exemplo: Altura de Adultos (cm)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Altura (cm)")
axes[1].set_ylabel("Densidade")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dist-normal.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: dist-normal.png")

# Propriedades da normal
print(f"  Média das amostras: {amostras.mean():.2f} (esperado: 170)")
print(f"  Desvio padrão: {amostras.std():.2f} (esperado: 8)")
print(f"  68% dos dados entre: [{170-8:.0f}, {170+8:.0f}]")
print(f"  95% dos dados entre: [{170-2*8:.0f}, {170+2*8:.0f}]")

# ============================================================
# 2. Distribuição Binomial
# ============================================================
print("\n[INFO] Gerando Distribuição Binomial...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PMF com diferentes parâmetros
for n, p, cor in [(20, 0.5, "#2563eb"), (20, 0.3, "#10b981"), (40, 0.7, "#ef4444")]:
    k = np.arange(0, n + 1)
    pmf = stats.binom.pmf(k, n, p)
    axes[0].bar(k, pmf, alpha=0.6, color=cor, label=f"n={n}, p={p}", edgecolor="white")

axes[0].set_title("Distribuição Binomial — PMF", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Número de Sucessos (k)")
axes[0].set_ylabel("Probabilidade")
axes[0].legend(fontsize=10)

# Exemplo prático: taxa de conversão
np.random.seed(42)
conversoes = np.random.binomial(n=100, p=0.05, size=1000)
axes[1].hist(conversoes, bins=range(0, 20), density=True, alpha=0.7,
             color="#2563eb", edgecolor="white", align="left",
             label="Simulação (1000 dias)")
k_plot = np.arange(0, 20)
axes[1].plot(k_plot, stats.binom.pmf(k_plot, 100, 0.05), "ro-", linewidth=2,
             markersize=5, label="PMF teórica (n=100, p=0.05)")
axes[1].set_title("Exemplo: Conversões diárias\n(100 visitantes, 5% de conversão)",
                   fontsize=12, fontweight="bold")
axes[1].set_xlabel("Número de Conversões")
axes[1].set_ylabel("Probabilidade")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dist-binomial.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: dist-binomial.png")

media_conv = stats.binom.mean(100, 0.05)
std_conv = stats.binom.std(100, 0.05)
print(f"  Conversões esperadas por dia: {media_conv:.1f}")
print(f"  Desvio padrão: {std_conv:.2f}")

# ============================================================
# 3. Distribuição de Poisson
# ============================================================
print("\n[INFO] Gerando Distribuição de Poisson...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PMF com diferentes lambdas
for lam, cor in [(2, "#2563eb"), (5, "#10b981"), (10, "#ef4444")]:
    k = np.arange(0, 25)
    pmf = stats.poisson.pmf(k, lam)
    axes[0].plot(k, pmf, "o-", color=cor, linewidth=2, markersize=5,
                 label=f"λ={lam}")

axes[0].set_title("Distribuição de Poisson — PMF", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Número de Eventos (k)")
axes[0].set_ylabel("Probabilidade")
axes[0].legend(fontsize=10)

# Exemplo prático: bugs por sprint
np.random.seed(42)
bugs = np.random.poisson(lam=4, size=500)
axes[1].hist(bugs, bins=range(0, 15), density=True, alpha=0.7,
             color="#2563eb", edgecolor="white", align="left",
             label="Simulação (500 sprints)")
k_plot = np.arange(0, 15)
axes[1].plot(k_plot, stats.poisson.pmf(k_plot, 4), "ro-", linewidth=2,
             markersize=5, label="PMF teórica (λ=4)")
axes[1].set_title("Exemplo: Bugs por Sprint\n(média de 4 bugs)",
                   fontsize=12, fontweight="bold")
axes[1].set_xlabel("Número de Bugs")
axes[1].set_ylabel("Probabilidade")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dist-poisson.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: dist-poisson.png")

prob_zero = stats.poisson.pmf(0, 4)
prob_8_plus = 1 - stats.poisson.cdf(7, 4)
print(f"  P(0 bugs na sprint) = {prob_zero:.4f} ({prob_zero:.1%})")
print(f"  P(8+ bugs na sprint) = {prob_8_plus:.4f} ({prob_8_plus:.1%})")

# ============================================================
# 4. Distribuição Exponencial
# ============================================================
print("\n[INFO] Gerando Distribuição Exponencial...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF com diferentes lambdas
x = np.linspace(0, 8, 500)
for lam, cor in [(0.5, "#2563eb"), (1.0, "#10b981"), (2.0, "#ef4444")]:
    y = stats.expon.pdf(x, scale=1/lam)
    axes[0].plot(x, y, linewidth=2.5, color=cor, label=f"λ={lam}")

axes[0].set_title("Distribuição Exponencial — PDF", fontsize=13, fontweight="bold")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Densidade de Probabilidade")
axes[0].legend(fontsize=10)

# Exemplo prático: tempo entre chamadas
np.random.seed(42)
tempo_chamadas = np.random.exponential(scale=5, size=2000)  # média de 5 min
axes[1].hist(tempo_chamadas, bins=50, density=True, alpha=0.7,
             color="#2563eb", edgecolor="white",
             label="Simulação (n=2000)")
x_fit = np.linspace(0, 30, 200)
axes[1].plot(x_fit, stats.expon.pdf(x_fit, scale=5), "r-", linewidth=2.5,
             label="PDF teórica (λ=0.2, média=5min)")
axes[1].set_title("Exemplo: Tempo entre Chamadas\n(call center, média=5 min)",
                   fontsize=12, fontweight="bold")
axes[1].set_xlabel("Tempo (minutos)")
axes[1].set_ylabel("Densidade")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dist-exponencial.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: dist-exponencial.png")

prob_10min = 1 - stats.expon.cdf(10, scale=5)
print(f"  P(esperar > 10 min) = {prob_10min:.4f} ({prob_10min:.1%})")

# ============================================================
# 5. Distribuição Uniforme
# ============================================================
print("\n[INFO] Gerando Distribuição Uniforme...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF
for a, b, cor in [(0, 1, "#2563eb"), (2, 6, "#10b981"), (-1, 3, "#ef4444")]:
    x = np.linspace(a - 1, b + 1, 500)
    y = stats.uniform.pdf(x, a, b - a)
    axes[0].plot(x, y, linewidth=2.5, color=cor, label=f"a={a}, b={b}")
    axes[0].fill_between(x, y, alpha=0.1, color=cor)

axes[0].set_title("Distribuição Uniforme — PDF", fontsize=13, fontweight="bold")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Densidade de Probabilidade")
axes[0].legend(fontsize=10)
axes[0].set_ylim(-0.05, 1.2)

# Exemplo prático: gerador de números aleatórios
np.random.seed(42)
aleatorios = np.random.uniform(0, 1, size=10000)
axes[1].hist(aleatorios, bins=50, density=True, alpha=0.7,
             color="#2563eb", edgecolor="white",
             label="Amostras (n=10000)")
axes[1].axhline(y=1, color="red", linewidth=2, linestyle="--",
                label="PDF teórica (f(x)=1)")
axes[1].set_title("Exemplo: Gerador Aleatório\n(U(0,1) — base do np.random)",
                   fontsize=12, fontweight="bold")
axes[1].set_xlabel("Valor")
axes[1].set_ylabel("Densidade")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dist-uniforme.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: dist-uniforme.png")

# ============================================================
# 6. Painel comparativo de todas as distribuições
# ============================================================
print("\n[INFO] Gerando painel comparativo...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
cores = ["#2563eb", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6"]

# Normal
x = np.linspace(-4, 4, 300)
axes[0, 0].plot(x, stats.norm.pdf(x), linewidth=2.5, color=cores[0])
axes[0, 0].fill_between(x, stats.norm.pdf(x), alpha=0.15, color=cores[0])
axes[0, 0].set_title("Normal\nμ=0, σ=1", fontweight="bold")
axes[0, 0].set_ylabel("Densidade")

# Binomial
k = np.arange(0, 21)
axes[0, 1].bar(k, stats.binom.pmf(k, 20, 0.5), color=cores[1], edgecolor="white")
axes[0, 1].set_title("Binomial\nn=20, p=0.5", fontweight="bold")
axes[0, 1].set_ylabel("Probabilidade")

# Poisson
k = np.arange(0, 20)
axes[0, 2].bar(k, stats.poisson.pmf(k, 5), color=cores[2], edgecolor="white")
axes[0, 2].set_title("Poisson\nλ=5", fontweight="bold")

# Exponencial
x = np.linspace(0, 6, 300)
axes[1, 0].plot(x, stats.expon.pdf(x), linewidth=2.5, color=cores[3])
axes[1, 0].fill_between(x, stats.expon.pdf(x), alpha=0.15, color=cores[3])
axes[1, 0].set_title("Exponencial\nλ=1", fontweight="bold")
axes[1, 0].set_ylabel("Densidade")
axes[1, 0].set_xlabel("x")

# Uniforme
x = np.linspace(-0.5, 1.5, 300)
axes[1, 1].plot(x, stats.uniform.pdf(x), linewidth=2.5, color=cores[4])
axes[1, 1].fill_between(x, stats.uniform.pdf(x), alpha=0.15, color=cores[4])
axes[1, 1].set_title("Uniforme\na=0, b=1", fontweight="bold")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylim(-0.05, 1.5)

# Tabela resumo
axes[1, 2].axis("off")
tabela_dados = [
    ["Normal", "Contínua", "μ, σ", "Alturas, notas"],
    ["Binomial", "Discreta", "n, p", "Conversões, defeitos"],
    ["Poisson", "Discreta", "λ", "Bugs, chamadas"],
    ["Exponencial", "Contínua", "λ", "Tempo entre eventos"],
    ["Uniforme", "Contínua", "a, b", "Sorteios, random"],
]
tabela = axes[1, 2].table(
    cellText=tabela_dados,
    colLabels=["Distribuição", "Tipo", "Parâmetros", "Exemplo"],
    loc="center",
    cellLoc="center",
)
tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1, 1.8)
axes[1, 2].set_title("Resumo", fontweight="bold")

plt.suptitle("Distribuições Estatísticas — Visão Geral",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dist-painel-comparativo.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("[INFO] Salvo: dist-painel-comparativo.png")

# Salvar métricas
resumo = {
    "normal_amostras_media": float(amostras.mean()),
    "normal_amostras_std": float(amostras.std()),
    "binomial_conversoes_media": float(media_conv),
    "poisson_prob_zero_bugs": float(prob_zero),
    "poisson_prob_8_plus_bugs": float(prob_8_plus),
    "exponencial_prob_10min": float(prob_10min),
}
with open(os.path.join(SAVE_DIR, "metricas.json"), "w") as f:
    json.dump(resumo, f, indent=2)

print("\n[INFO] Concluído!")
