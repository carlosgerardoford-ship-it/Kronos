import os
import sys
import json
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

OUTPUT_DIR = '../outputs'
SYMBOL = 'BTCUSDT'
# usage: python prediction_btc.py [interval=1h] [lookback=400] [pred_len=24] [forecast]
# default: backtest (the last pred_len candles are held out as ground truth)
# 'forecast': use the latest lookback candles and predict the future
INTERVAL = sys.argv[1] if len(sys.argv) > 1 else '1h'
LOOKBACK = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PRED_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 24
FORECAST = len(sys.argv) > 4 and sys.argv[4] == 'forecast'
PREFIX = f'btc_{INTERVAL}_forecast' if FORECAST else f'btc_{INTERVAL}'


def fetch_klines(symbol, interval, limit):
    # Last `limit` closed candles (the still-open candle is dropped)
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit + 1}"
    with urllib.request.urlopen(url, timeout=30) as r:
        rows = json.load(r)[:-1]
    df = pd.DataFrame(rows).iloc[:, :8]
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'amount']
    df['timestamps'] = pd.to_datetime(df['open_time'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[c] = df[c].astype(float)
    return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]


os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Data: LOOKBACK candles as context (+ PRED_LEN candles as ground truth in backtest)
df = fetch_klines(SYMBOL, INTERVAL, LOOKBACK if FORECAST else LOOKBACK + PRED_LEN)
df.to_csv(os.path.join(OUTPUT_DIR, f'{PREFIX}_input.csv'), index=False)

# 2. Model (CPU)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

# 3. Predict
cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
x_df = df.loc[:LOOKBACK - 1, cols]
x_timestamp = df.loc[:LOOKBACK - 1, 'timestamps']
if FORECAST:
    step = pd.Timedelta(INTERVAL)
    y_timestamp = pd.Series(pd.date_range(x_timestamp.iloc[-1] + step, periods=PRED_LEN, freq=step))
else:
    y_timestamp = df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1, 'timestamps']

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
pred_df.to_csv(os.path.join(OUTPUT_DIR, f'{PREFIX}_predictions.csv'), index_label='timestamps')
print(pred_df.head())

# 4. Plot (last 100 context candles + ground truth vs prediction)
hist = df.loc[LOOKBACK - 100:LOOKBACK - 1].set_index('timestamps')
truth = None if FORECAST else df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1].set_index('timestamps')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(hist['close'], color='blue', label='History')
if truth is not None:
    ax1.plot(truth['close'], color='blue', linestyle='--', label='Ground Truth')
ax1.plot(pred_df['close'], color='red', label='Prediction')
ax1.set_ylabel('BTC/USDT Close')
ax1.legend(loc='best')
ax1.grid(True)
ax2.plot(hist['volume'], color='blue')
if truth is not None:
    ax2.plot(truth['volume'], color='blue', linestyle='--')
ax2.plot(pred_df['volume'], color='red')
ax2.set_ylabel('Volume (BTC)')
ax2.grid(True)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{PREFIX}_prediction.png'), dpi=150)
