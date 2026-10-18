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
LOOKBACK = 400
PRED_LEN = 24


def fetch_klines(symbol, limit):
    # Last `limit` closed 1h candles (the still-open candle is dropped)
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit={limit + 1}"
    with urllib.request.urlopen(url, timeout=30) as r:
        rows = json.load(r)[:-1]
    df = pd.DataFrame(rows).iloc[:, :8]
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'amount']
    df['timestamps'] = pd.to_datetime(df['open_time'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[c] = df[c].astype(float)
    return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]


os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Data: LOOKBACK candles as context + PRED_LEN candles as ground truth
df = fetch_klines(SYMBOL, LOOKBACK + PRED_LEN)
df.to_csv(os.path.join(OUTPUT_DIR, 'btc_1h_input.csv'), index=False)

# 2. Model (CPU)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

# 3. Predict
cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
x_df = df.loc[:LOOKBACK - 1, cols]
x_timestamp = df.loc[:LOOKBACK - 1, 'timestamps']
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
pred_df.to_csv(os.path.join(OUTPUT_DIR, 'btc_1h_predictions.csv'), index_label='timestamps')
print(pred_df.head())

# 4. Plot (last 100 context candles + ground truth vs prediction)
hist = df.loc[LOOKBACK - 100:LOOKBACK - 1].set_index('timestamps')
truth = df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1].set_index('timestamps')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(hist['close'], color='blue', label='History')
ax1.plot(truth['close'], color='blue', linestyle='--', label='Ground Truth')
ax1.plot(pred_df['close'], color='red', label='Prediction')
ax1.set_ylabel('BTC/USDT Close')
ax1.legend(loc='best')
ax1.grid(True)
ax2.plot(hist['volume'], color='blue')
ax2.plot(truth['volume'], color='blue', linestyle='--')
ax2.plot(pred_df['volume'], color='red')
ax2.set_ylabel('Volume (BTC)')
ax2.grid(True)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'btc_1h_prediction.png'), dpi=150)
