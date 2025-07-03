import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV dosyasını oku
df = pd.read_csv("outputs/results/equity_curve.csv")  # dosya yolunu gerektiğinde değiştir

# Portföy değeri
equity = df["portfolio_value"].values

# Getiri hesapla
returns = np.diff(equity) / equity[:-1]

# Metrik hesapları
initial_value = equity[0]
final_value = equity[-1]
net_profit = final_value - initial_value
return_pct = (final_value / initial_value - 1) * 100
sharpe = returns.mean() / (returns.std() + 1e-8)
cummax = np.maximum.accumulate(equity)
drawdown = (cummax - equity) / cummax
max_drawdown = drawdown.max() * 100

# Metin çıktısı
print("📊 EQUITY ANALYSIS")
print(f"Initial Value     : {initial_value:.2f} USDT")
print(f"Final Value       : {final_value:.2f} USDT")
print(f"Net Profit        : {net_profit:.2f} USDT")
print(f"Return (%)        : {return_pct:.2f}%")
print(f"Sharpe Ratio      : {sharpe:.4f}")
print(f"Max Drawdown (%)  : {max_drawdown:.2f}%")

# Grafik: equity curve
plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y=equity, label="Portfolio Value")
plt.title("Equity Curve")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value (USDT)")
plt.grid(True)
plt.tight_layout()
plt.show()
