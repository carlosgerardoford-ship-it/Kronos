import os
import sys
import json
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

OUTPUT_DIR = '../outputs'
TZ = 'America/New_York'
# usage: python prediction_stock.py [symbol=MSTR] [interval=1h|1d] [lookback=400] [pred_len=24] [forecast]
# default: backtest (the last pred_len candles are held out as ground truth)
# 'forecast': use the latest lookback candles and predict the future (regular trading hours only)
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else 'MSTR'
INTERVAL = sys.argv[2] if len(sys.argv) > 2 else '1h'
LOOKBACK = int(sys.argv[3]) if len(sys.argv) > 3 else 400
PRED_LEN = int(sys.argv[4]) if len(sys.argv) > 4 else 24
FORECAST = len(sys.argv) > 5 and sys.argv[5] == 'forecast'
PREFIX = f'{SYMBOL.lower()}_{INTERVAL}' + ('_forecast' if FORECAST else '')
RANGE = {'1h': '1y', '1d': '5y'}[INTERVAL]


def fetch_candles(symbol, interval, limit):
    # Yahoo Finance candles in New York local time; the still-open candle is dropped
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={RANGE}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)['chart']['result'][0]
    q = res['indicators']['quote'][0]
    start = pd.to_datetime(res['timestamp'], unit='s', utc=True).tz_convert(TZ)
    df = pd.DataFrame({'start': start, **{c: q[c] for c in ['open', 'high', 'low', 'close', 'volume']}}).dropna()
    df = df[df['volume'] > 0]  # Yahoo adds a zero-volume placeholder candle for the last session
    # candle end: +1h for hourly, 16:00 for daily
    end = df['start'] + pd.Timedelta('1h') if interval == '1h' else df['start'].dt.normalize() + pd.Timedelta(hours=16)
    df = df[end <= pd.Timestamp.now(tz=TZ)]
    df['timestamps'] = df['start'].dt.tz_localize(None)
    if interval == '1d':
        df['timestamps'] = df['timestamps'].dt.normalize()
    df['amount'] = df['close'] * df['volume']  # Yahoo has no turnover: approximated
    return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].tail(limit).reset_index(drop=True)


def future_timestamps(last, interval, n):
    # Next n regular-session candle starts after `last` (Mon-Fri, holidays not excluded)
    if interval == '1d':
        return pd.Series(pd.bdate_range(last.normalize() + pd.Timedelta(days=1), periods=n))
    days = pd.bdate_range(last.normalize(), periods=n // 7 + 3)
    slots = [d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h) for d in days for h in range(7)]
    return pd.Series([t for t in slots if t > last][:n])


os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Data: LOOKBACK candles as context (+ PRED_LEN candles as ground truth in backtest)
df = fetch_candles(SYMBOL, INTERVAL, LOOKBACK if FORECAST else LOOKBACK + PRED_LEN)
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
    y_timestamp = future_timestamps(x_timestamp.iloc[-1], INTERVAL, PRED_LEN)
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

# 4. Plot (last 100 context candles + prediction, plus ground truth in backtest); x = candle index, not clock time
hist = df.loc[max(LOOKBACK - 100, 0):LOOKBACK - 1]
n_h = len(hist)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
xs_h = range(n_h)
xs_p = range(n_h, n_h + PRED_LEN)
ax1.plot(xs_h, hist['close'], color='blue', label='History')
ax2.plot(xs_h, hist['volume'], color='blue')
if not FORECAST:
    truth = df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1]
    ax1.plot(xs_p, truth['close'], color='blue', linestyle='--', label='Ground Truth')
    ax2.plot(xs_p, truth['volume'], color='blue', linestyle='--')
ax1.plot(xs_p, pred_df['close'], color='red', label='Prediction')
ax2.plot(xs_p, pred_df['volume'], color='red')
ax1.set_ylabel(f'{SYMBOL} Close (USD)')
ax1.legend(loc='best')
ax1.grid(True)
ax2.set_ylabel('Volume (shares)')
ax2.grid(True)
ticks = list(range(0, n_h, 20)) + [n_h + PRED_LEN - 1]
labels = [hist['timestamps'].iloc[i].strftime('%m-%d %H:%M') for i in range(0, n_h, 20)] + [pred_df.index[-1].strftime('%m-%d %H:%M')]
ax2.set_xticks(ticks)
ax2.set_xticklabels(labels, rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{PREFIX}_prediction.png'), dpi=150)
