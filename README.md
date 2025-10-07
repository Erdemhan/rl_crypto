# ?? Crypto PPO Trading Bot

This project implements a crypto trading bot using Proximal Policy Optimization (PPO) in a custom Gym environment. It operates on the top 10 cryptocurrencies by trading volume and uses 7 technical indicators to form the observation space. The agent learns to make discrete decisions (Buy[i], Sell, Hold) to maximize risk-adjusted returns using cumulative profit, Sharpe ratio, and maximum drawdown as reward components.

## ?? Features

* PPO-based reinforcement learning agent (Actor-Critic)
* Custom Gym-compatible environment
* Discrete action space: Hold, Sell, Buy[1]-Buy[10]
* 10 coins x 7 technical indicators: RSI, MACD, ADX, ATR, Bollinger Width, OBV, Stochastic
* Reward combines profit, Sharpe Ratio, and Max Drawdown
* Live / paper trading support
* Fully modular and OOP-structured Python code

## ?? Installation

```bash
git clone https://github.com/erdmhn/crypto.git
cd crypto
pip install -r requirements.txt
```

## ??? Project Structure

* `src/crypto_rl/` - moduler cekirdek paket (config, env, agent, trainer, evaluation, pipelines)
* `data/` - data loading, indicator calculation, and dataset splitting
* `models/` - Actor and Critic neural network models
* `trainer/` - legacy wrapper for the new training loop
* `evaluation/` - legacy wrapper for backtesting utilities
* `scripts/` - CLI entry points (`train.py`, `test.py`, and `live.py`)
* `config/` - training and environment parameters in `config.yaml`
* `outputs/` - saved models, logs, equity curves

## 🧭 How the System Works

Yeni başlayanlara yönelik olarak, aşağıdaki adımlar sistemin uçtan uca nasıl çalıştığını anlatır:

1. **Koşuyu yapılandır**  
   `configs/config.yaml` dosyasını düzenleyin. `globals` bloğu varsayılan veri aralıkları, PPO hiperparametreleri, ödül katsayıları ve cihaz ayarlarını içerir; profiller (aggressive, balanced, defensive) yalnızca farklılık olan alanları override eder. Eğitim pipeline’ı `data/processed/coin_data.parquet` dosyasını salt-okur.
2. **Ajanı eğit (`python scripts/train.py --run-id 20251007`)**  
   CLI yapılandırmayı çözümler, `outputs/<run-id>_<profil>/` altında izole bir klasör oluşturur, rastgelelik kaynaklarını sabitleyip veriyi eğitim/doğrulama/test aralıklarına böler. PPO trainer her epoch `CryptoTradingEnv` üzerinden rollout toplar, GAE (lambda) ile avantajları hesaplar, aktör/kritik ağlarını günceller ve loglara yazar. `training.validate_every` > 0 olduğunda doğrulama setinde test yapılır; `training.best_metric` (varsayılan `net_profit`) iyileştiğinde o andaki ağırlıklar “en iyi” olarak saklanır. Eğitim sonunda bu snapshot `outputs/<run>/models/best_model.pth` dosyasına ve zaman damgalı yedeğe kaydedilir; loglar ve doğrulama CSV’leri aynı run klasörünün `logs/` ve `validation/` altındadır.
3. **Modele geri test yap (`python scripts/test.py --run-id 20251007`)**  
   Backtest pipeline’ı profillerin `best_model.pth` dosyalarını yükler, test aralığında deterministik (isterseniz rastgele) bir simülasyon koşturur ve çıktıları `outputs/<run>/results/` altına bırakır: `equity_curve.csv` portföy değerini, `trades_log.csv` gerçekleşen işlemleri, `actions.csv` ise her adımda alınan HOLD/BUY/SELL kararlarını ve geçersiz/tekrarlı aksiyon bayraklarını içerir.
4. **Canlı veya kâğıt üzerinde dene (`python scripts/live.py`)**  
   Bu betik aynı `best_model.pth` dosyasını yükler, gözlem vektörlerini (şimdilik offline veriden simüle ediliyor) gerçek zamanlı besler ve portföy değerini loglar. Gerçek borsa entegrasyonu için yalnızca veri akışını değiştirmek gerekir.

Her run dizini böylece kendi içinde tam döngüyü barındırır: ham veri `data/` dizininde kalır, deneylerin log/model çıktıları `outputs/<run-id>_<profil>/` altında toplanır ve en güncel şampiyon model her zaman `models/best_model.pth` dosyasında bulunur.

## ?? Training

```bash
python scripts/train.py
```

* Trains a PPO agent on historical hourly OHLCV data.
* Uses technical indicators per coin.
* Saves best model to: `outputs/models/best_model.pth`

## ??? Backtesting

```bash
python scripts/test.py
```

* Runs evaluation on a separate test set.
* Outputs:

  * `outputs/results/trades_log.csv`
  * `outputs/results/equity_curve.csv`

## ?? Performance Evaluation

```python
from evaluation.metrics import print_metrics
print_metrics("outputs/results/equity_curve.csv")
```

Outputs:

* Cumulative Return
* Sharpe Ratio
* Max Drawdown

## ?? Live / Paper Trading

```bash
python scripts/live.py
```

Simulates live execution (can be adapted for Binance API):

* Feeds current market state to trained agent
* Logs trades and portfolio value in real time

## ?? Configuration

All behavior is controlled via `config/config.yaml`, including:

* coin list and data intervals
* PPO hyperparameters
* environment and reward settings
* logging & model saving paths

## ?? License

MIT License

## ?? Author

Developed by [@erdmhn](https://github.com/erdmhn)
