"""Walk-forward validation of Kronos-small against the "price stays the same" baseline.

usage: python validate.py KO TLT BTCUSDT --interval 1h --lookback 400 --pred-len 24 --windows 20
       python validate.py KO --interval 1h --windows 40 --offset 480
       python validate.py KO --null-test
"""
import os
import sys
import json
import time
import argparse
import urllib.error
import urllib.request
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TZ = "America/New_York"
COLS = ["open", "high", "low", "close", "volume", "amount"]
CRYPTO_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD")
INTERVALS = ["5m", "15m", "1h", "4h", "1d"]
YAHOO_RANGE = {"5m": "60d", "15m": "60d", "1h": "2y", "1d": "5y"}


# ---------------------------------------------------------------- data

def is_crypto(ticker):
    return "/" in ticker or ticker.upper().endswith(CRYPTO_QUOTES)


def _get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_crypto(ticker, interval, need):
    """Last `need` closed Binance spot candles, paginating backwards (max 1000 per request)."""
    pair = ticker.upper().replace("/", "").replace("-", "")
    now_ms = int(time.time() * 1000)
    rows, end_time = [], None
    while len(rows) < need:
        limit = min(1000, need - len(rows) + (1 if end_time is None else 0))
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={pair}&interval={interval}&limit={limit}"
        if end_time is not None:
            url += f"&endTime={end_time}"
        try:
            chunk = _get_json(url)
        except urllib.error.HTTPError as e:
            raise ValueError(f"Binance no devolvió datos para {pair} (HTTP {e.code})")
        if end_time is None:
            chunk = [r for r in chunk if r[6] <= now_ms]  # drop the still-open candle
        if not chunk:
            break
        rows = chunk + rows
        end_time = chunk[0][0] - 1
    if not rows:
        raise ValueError(f"Binance no devolvió velas para {pair}")
    df = pd.DataFrame(rows).iloc[:, :8]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "amount"]
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in COLS:
        df[c] = df[c].astype(float)
    return df[["timestamps"] + COLS].tail(need).reset_index(drop=True)


def fetch_stock(ticker, interval):
    """All closed Yahoo Finance candles in the interval's range (New York local time)."""
    if interval not in YAHOO_RANGE:
        raise ValueError(f"Yahoo Finance no ofrece velas de {interval} para acciones (usar 5m, 15m, 1h o 1d)")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.upper()}"
           f"?interval={interval}&range={YAHOO_RANGE[interval]}")
    try:
        res = _get_json(url, {"User-Agent": "Mozilla/5.0"})["chart"]["result"][0]
    except urllib.error.HTTPError as e:
        raise ValueError(f"Yahoo Finance no devolvió datos para {ticker.upper()} (HTTP {e.code})")
    q = res["indicators"]["quote"][0]
    start = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(TZ)
    df = pd.DataFrame({"start": start, **{c: q[c] for c in ["open", "high", "low", "close", "volume"]}}).dropna()
    df = df[df["volume"] > 0]
    if interval == "1d":
        end = df["start"].dt.normalize() + pd.Timedelta(hours=16)
    else:
        end = df["start"] + pd.Timedelta(interval)
    df = df[end <= pd.Timestamp.now(tz=TZ)]
    df["timestamps"] = df["start"].dt.tz_localize(None)
    if interval == "1d":
        df["timestamps"] = df["timestamps"].dt.normalize()
    df["amount"] = df["close"] * df["volume"]  # Yahoo has no turnover: approximated
    return df[["timestamps"] + COLS].reset_index(drop=True)


def load_candles(ticker, interval, lookback, pred_len, windows, offset):
    need = lookback + windows * pred_len + offset
    df = fetch_crypto(ticker, interval, need) if is_crypto(ticker) else fetch_stock(ticker, interval)
    if len(df) < need:
        raise ValueError(f"Solo hay {len(df)} velas y se necesitan {need}")
    df = df.tail(need)
    if offset:
        df = df.iloc[:-offset]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- metrics

def best_lag(pred, truth, last, max_shift):
    # k > 0: la predicción llega tarde; k < 0: se adelanta. Empates → menor |k|.
    a, b = pred - last, truth - last
    n = len(a)
    best_k, best_err = 0, None
    for k in sorted(range(-max_shift, max_shift + 1), key=abs):
        if abs(k) >= n:
            continue
        err = abs(a[k:] - b[:n - k]).mean() if k >= 0 else abs(a[:n + k] - b[-k:]).mean()
        if best_err is None or err < best_err:
            best_k, best_err = k, err
    return best_k, best_err


def best_lag_corr(pred, truth, last, max_shift):
    """Lag by correlation of candle-to-candle changes (no edge pile-up, unlike best_lag).

    Returns (best_k, corr at best_k, corr at 0); k > 0: the prediction arrives late. The search range
    should stay well under the path length so the overlap is never tiny (e.g. pred_len // 4).
    """
    a = np.diff(np.concatenate([[last], pred]))
    b = np.diff(np.concatenate([[last], truth]))
    n, corr = len(a), {}
    for k in range(-max_shift, max_shift + 1):
        x, y = (a[k:], b[:n - k]) if k >= 0 else (a[:n + k], b[-k:])
        if len(x) >= 3 and x.std() > 0 and y.std() > 0:
            corr[k] = float(np.corrcoef(x, y)[0, 1])
    if not corr:
        return 0, float("nan"), float("nan")
    best = max(sorted(corr, key=abs), key=lambda k: corr[k])
    return best, corr[best], corr.get(0, float("nan"))


def wilson_interval(hits, n, z=1.96):
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return centre - half, centre + half


def window_metrics(ctx_close_last, pred_close, real_close, pred_len):
    last = ctx_close_last
    pred_end, real_end = pred_close[-1], real_close[-1]
    lag, mae_at_lag = best_lag(pred_close, real_close, last, max(1, pred_len // 3))
    return {
        "last_close": last, "pred_end": pred_end, "real_end": real_end,
        "hit": bool((real_end > last) == (pred_end > last)), "real_up": bool(real_end > last),
        "mae": float(np.abs(real_close - pred_close).mean()),
        "mae_baseline": float(np.abs(real_close - last).mean()),
        "lag": int(lag), "mae_at_lag": float(mae_at_lag),
    }


def ticker_stats(rows):
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    ups = sum(r["real_up"] for r in rows)
    lags = np.array([r["lag"] for r in rows])
    lo, hi = wilson_interval(hits, n)
    return {
        "n": n, "hits": hits, "pct": hits / n, "lo": lo, "hi": hi, "ups": ups, "base": max(ups, n - ups) / n,
        "mae": float(np.mean([r["mae"] for r in rows])),
        "mae_baseline": float(np.mean([r["mae_baseline"] for r in rows])),
        "beats": sum(r["mae"] < r["mae_baseline"] for r in rows),
        "mae_at_lag": float(np.mean([r["mae_at_lag"] for r in rows])),
        "lag_abs": float(np.abs(lags).mean()), "lag_mean": float(lags.mean()), "lag_median": float(np.median(lags)),
        "late": int((lags > 0).sum()), "early": int((lags < 0).sum()), "zero": int((lags == 0).sum()),
    }


def _num(x):
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"


def format_summary(ticker, interval, pred_len, samples, periods, s):
    return "\n".join([
        f"{ticker} · {interval} · {s['n']} ventanas",
        f"Cada una predice {pred_len} velas, con {samples} muestra(s)",
        f"Período: {periods[0]} → {periods[1]}",
        f"Dirección acertada:   {s['hits']} de {s['n']} ({s['pct'] * 100:.0f}%)",
        f"  Margen de error 95%: {s['lo'] * 100:.0f}% a {s['hi'] * 100:.0f}%",
        f"  El precio subió en {s['ups']} de {s['n']} ventanas; siempre apostar a la más frecuente "
        f"acertaría {s['base'] * 100:.0f}%",
        f"Error abs. medio:     {s['mae']:.2f}",
        f"Baseline \"se queda igual\": {s['mae_baseline']:.2f}",
        f"Le gana al baseline:  {s['beats']} de {s['n']} ventanas",
        f"Desfase absoluto medio: {s['lag_abs']:.1f} velas (sin importar el signo)",
        f"  Signo medio: {s['lag_mean']:.1f}   mediana: {_num(s['lag_median'])}",
        f"  Tarde: {s['late']}   Adelantado: {s['early']}   Sin desfase: {s['zero']}",
        f"  Error alineando por desfase: {s['mae_at_lag']:.2f} (contra {s['mae']:.2f} sin alinear)",
        "  (+ = la predicción llega tarde; - = se adelanta)",
    ])


def format_table(entries):
    head = ["ticker", "aciertos", "%", "IC95", "base", "err medio", "baseline", "gana base.", "desf. abs", "tarde/adel."]
    lines = []
    for ticker, s, err in entries:
        if err:
            lines.append([ticker, f"ERROR: {err}"])
            continue
        lines.append([ticker, f"{s['hits']}/{s['n']}", f"{s['pct'] * 100:.0f}%",
                      f"{s['lo'] * 100:.0f}-{s['hi'] * 100:.0f}%", f"{s['base'] * 100:.0f}%",
                      f"{s['mae']:.2f}", f"{s['mae_baseline']:.2f}", f"{s['beats']}/{s['n']}",
                      f"{s['lag_abs']:.1f}", f"{s['late']}/{s['early']}"])
    full = [l for l in lines if len(l) == len(head)]
    widths = [max([len(head[i])] + [len(l[i]) for l in full]) for i in range(len(head))]
    fmt = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
    out = [fmt(head), fmt(["-" * w for w in widths])]
    out += [fmt(l) if len(l) == len(head) else f"{l[0].ljust(widths[0])}  {l[1]}" for l in lines]
    return "\n".join(out)


# ---------------------------------------------------------------- null test

def _null_lags(rows, max_shift):
    """Best lag pairing prediction i with the real path of window j != i (each re-anchored to its own last)."""
    out = []
    for i, ri in enumerate(rows):
        for j, rj in enumerate(rows):
            if i != j:
                out.append(best_lag(ri["pred_path"] - ri["last_close"], rj["real_path"] - rj["last_close"],
                                    0.0, max_shift)[0])
    return np.array(out)


def lag_summary(lags, max_shift):
    return {"n": len(lags), "abs": float(np.abs(lags).mean()), "early": float((lags < 0).mean()),
            "late": float((lags > 0).mean()), "zero": float((lags == 0).mean()),
            "edge": float((np.abs(lags) == max_shift).mean())}


def format_null(ticker, rows, pred_len):
    max_shift = max(1, pred_len // 3)
    m = lag_summary(np.array([r["lag"] for r in rows]), max_shift)
    nl = lag_summary(_null_lags(rows, max_shift), max_shift)
    artifact = abs(m["abs"] - nl["abs"]) < 1.0 and abs(m["early"] - nl["early"]) < 0.10
    line = lambda label, x: (f"  {label} ({x['n']:>4}): abs {x['abs']:.1f} | adelantado {x['early'] * 100:.0f}% | "
                             f"tarde {x['late'] * 100:.0f}% | sin desfase {x['zero'] * 100:.0f}% | "
                             f"en el borde {x['edge'] * 100:.0f}%")
    return "\n".join([
        f"{ticker} · prueba sin habilidad (desfase máx. {max_shift})",
        line("Pares correctos   ", m),
        line("Pares sin relación", nl),
        "  → " + ("parecidos: el desfase medido es probablemente un artefacto de la medida"
                  if artifact else "distintos: el emparejamiento correcto se comporta distinto al azar"),
    ])


# ---------------------------------------------------------------- volatility

def vol_metrics(ctx_tail, pred, truth):
    """Range (high-low) and close-to-close volatility of the predicted/real window and of the last context
    window (the naive "same volatility as before" baseline), all as a fraction of the last context close."""
    last = float(ctx_tail["close"].iloc[-1])
    rng = lambda d: (float(d["high"].max()) - float(d["low"].min())) / last
    vol = lambda c: float(np.std(np.diff(np.asarray(c, dtype=float)))) / last
    ctx_prev = ctx_tail.iloc[1:]  # last P candles of the context
    return {"pred_range": rng(pred), "real_range": rng(truth), "ctx_range": rng(ctx_prev),
            "pred_vol": vol(np.concatenate([[last], pred["close"]])),
            "real_vol": vol(np.concatenate([[last], truth["close"]])), "ctx_vol": vol(ctx_tail["close"])}


def vol_stats(rows):
    out = {}
    for key in ("range", "vol"):
        pr, rr, cr = (np.maximum(np.array([r[f"{k}_{key}"] for r in rows]), 1e-9) for k in ("pred", "real", "ctx"))
        sp = lambda x, y: float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])  # Spearman, no scipy
        out[key] = {"corr_model": sp(pr, rr), "corr_base": sp(cr, rr),
                    "err_model": float(np.median(np.abs(np.log(pr / rr)))),
                    "err_base": float(np.median(np.abs(np.log(cr / rr)))),
                    "ratio": float(np.median(pr / rr)), "n": len(rows)}
    return out


def format_vol(ticker, rows):
    s = vol_stats(rows)
    names = {"range": "Rango (máx-mín) de la ventana", "vol": "Volatilidad (desv. de cambios de cierre)"}
    lines = [f"{ticker} · volatilidad ({s['range']['n']} ventanas; baseline = misma volatilidad que en el contexto)"]
    for key in ("range", "vol"):
        x = s[key]
        lines += [f"{names[key]}",
                  f"  Correlación (Spearman) con el real:  modelo {x['corr_model']:+.2f} | baseline {x['corr_base']:+.2f}",
                  f"  Error log mediano |ln(pred/real)|:   modelo {x['err_model']:.2f} | baseline {x['err_base']:.2f}",
                  f"  Mediana predicho/real: {x['ratio']:.2f}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- walk-forward

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        sys.path.append(ROOT_DIR)
        from model import Kronos, KronosTokenizer, KronosPredictor
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        _predictor = KronosPredictor(model, tokenizer, max_context=512)
    return _predictor


def run_ticker(ticker, args, checkpoint=None):
    import torch
    df = load_candles(ticker, args.interval, args.lookback, args.pred_len, args.windows, args.offset)
    predictor = get_predictor()
    P, L = args.pred_len, args.lookback
    rows = []
    for i in range(args.windows):
        print(f"{ticker} · ventana {i + 1} de {args.windows}", flush=True)
        ctx = df.iloc[i * P:i * P + L]
        truth = df.iloc[i * P + L:i * P + L + P]
        torch.manual_seed(args.seed + i)
        pred = predictor.predict(
            df=ctx[COLS].reset_index(drop=True),
            x_timestamp=ctx["timestamps"].reset_index(drop=True),
            y_timestamp=truth["timestamps"].reset_index(drop=True),
            pred_len=P, T=1.0, top_p=0.9, sample_count=args.samples, verbose=False)
        pred_path = pred["close"].to_numpy(dtype=float)
        real_path = truth["close"].to_numpy(dtype=float)
        m = window_metrics(float(ctx["close"].iloc[-1]), pred_path, real_path, P)
        m.update(window=i, ctx_end=ctx["timestamps"].iloc[-1], pred_path=pred_path, real_path=real_path)
        m.update(vol_metrics(ctx.tail(P + 1), pred, truth))
        rows.append(m)
        if checkpoint and (i + 1) % 10 == 0:
            save_csv(checkpoint, rows)
    return rows


def save_csv(path, rows):
    cols = ["window", "ctx_end", "last_close", "pred_end", "real_end", "hit", "real_up", "mae", "mae_baseline",
            "lag", "mae_at_lag", "pred_range", "real_range", "ctx_range", "pred_vol", "real_vol", "ctx_vol"]
    out = pd.DataFrame([{c: r[c] for c in cols} for r in rows])
    out["close_pred"] = [";".join(f"{v:.6g}" for v in r["pred_path"]) for r in rows]
    out["close_real"] = [";".join(f"{v:.6g}" for v in r["real_path"]) for r in rows]
    out.to_csv(path, index=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="+")
    p.add_argument("--interval", choices=INTERVALS, default="1h")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--windows", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--out", default="outputs/validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--null-test", action="store_true")
    a = p.parse_args(argv)
    for name, lo, hi in [("lookback", 64, 512), ("pred_len", 1, 240), ("samples", 1, 20), ("windows", 2, 200)]:
        if not lo <= getattr(a, name) <= hi:
            p.error(f"--{name.replace('_', '-')} debe estar entre {lo} y {hi}")
    if a.offset < 0:
        p.error("--offset debe ser >= 0")
    return a


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.interval}" + (f"_off{args.offset}" if args.offset else "")
    entries, texts = [], []
    for ticker in args.tickers:
        name = ticker.upper().replace("/", "")
        try:
            rows = run_ticker(ticker, args, os.path.join(args.out, f"{name}_{tag}.csv"))
        except Exception as e:
            print(f"{name}: ERROR: {e}", flush=True)
            entries.append((name, None, str(e)))
            continue
        s = ticker_stats(rows)
        periods = (rows[0]["ctx_end"].strftime("%Y-%m-%d %H:%M"), rows[-1]["ctx_end"].strftime("%Y-%m-%d %H:%M"))
        text = format_summary(name, args.interval, args.pred_len, args.samples, periods, s)
        text += "\n\n" + format_vol(name, rows)
        if args.null_test:
            text += "\n\n" + format_null(name, rows, args.pred_len)
        print("\n" + text + "\n", flush=True)
        texts.append(text)
        save_csv(os.path.join(args.out, f"{name}_{tag}.csv"), rows)
        entries.append((name, s, None))
    table = format_table(entries)
    print(table)
    with open(os.path.join(args.out, f"summary_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts) + "\n\n" + table + "\n")


if __name__ == "__main__":
    main()
