"""Model-free tests for validate.py. Run: python test_validate.py (or pytest)."""
import numpy as np
import pandas as pd
import validate as v


def shifted(truth, k):
    # k > 0: prediction arrives late (pred[t + k] = truth[t]); k < 0: it leads
    if k >= 0:
        return np.concatenate([np.full(k, truth[0]), truth[:len(truth) - k]])
    return np.concatenate([truth[-k:], np.full(-k, truth[-1])])


def test_best_lag_exact():
    truth = 100 + np.cumsum(np.random.default_rng(1).normal(size=60))
    for k in (3, -3, 8, -8, 0):
        got, err = v.best_lag(shifted(truth, k), truth, truth[0], max_shift=10)
        assert got == k and err < 1e-9, (k, got, err)


def test_best_lag_default_range():
    truth = 100 + np.cumsum(np.random.default_rng(2).normal(size=24))
    assert v.best_lag(shifted(truth, 8), truth, truth[0], max_shift=8)[0] == 8


def test_best_lag_short_path():
    k, err = v.best_lag(np.array([1.0]), np.array([2.0]), 1.0, 1)
    assert k == 0 and err == 1.0


def test_wilson():
    lo, hi = v.wilson_interval(8, 20)
    assert round(lo * 100) == 22 and round(hi * 100) == 61, (lo, hi)
    lo, hi = v.wilson_interval(20, 20)
    assert hi > 0.99 and lo > 0.8


def test_window_metrics():
    real = np.array([10.0, 11.0, 12.0])
    m = v.window_metrics(10.0, np.array([10.5, 11.5, 11.0]), real, 3)
    assert m["hit"] and m["real_up"]
    assert abs(m["mae"] - np.abs(real - np.array([10.5, 11.5, 11.0])).mean()) < 1e-12
    assert abs(m["mae_baseline"] - 1.0) < 1e-12


def fake_rows(n=20, p=24, skill=False, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        real = 100 + np.cumsum(rng.normal(size=p))
        pred = shifted(real, 2) if skill else 100 + np.cumsum(rng.normal(size=p))
        m = v.window_metrics(100.0, pred, real, p)
        m.update(window=i, ctx_end=pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i), pred_path=pred, real_path=real)
        rows.append(m)
    return rows


def test_summary_and_table_format():
    s = v.ticker_stats(fake_rows())
    txt = v.format_summary("KO", "1h", 24, 1, ("a", "b"), s)
    lines = txt.split("\n")
    assert lines[0] == "KO · 1h · 20 ventanas"
    assert lines[1] == "Cada una predice 24 velas, con 1 muestra(s)"
    assert lines[3].startswith("Dirección acertada:   ") and " de 20 (" in lines[3]
    assert lines[4].startswith("  Margen de error 95%: ") and lines[4].endswith("%")
    assert "siempre apostar a la más frecuente acertaría" in lines[5]
    assert lines[7].startswith("Baseline \"se queda igual\": ")
    assert lines[-1] == "  (+ = la predicción llega tarde; - = se adelanta)"
    table = v.format_table([("KO", s, None), ("ZZZ", None, "Yahoo Finance no devolvió datos para ZZZ (HTTP 404)")])
    assert "KO" in table and "ERROR: Yahoo" in table


def test_null_test_separates_skill_from_noise():
    rows = fake_rows(skill=True)
    ms = 8
    matched = v.lag_summary(np.array([r["lag"] for r in rows]), ms)
    null = v.lag_summary(v._null_lags(rows, ms), ms)
    assert matched["late"] > 0.8 and matched["abs"] < 2.5      # real skill: consistent lag of 2
    assert null["abs"] > matched["abs"]                         # unrelated pairs: no consistent lag
    assert "prueba sin habilidad" in v.format_null("X", rows, 24)


def test_windows_no_leak_and_offset(monkeypatch=None):
    n = 400 + 5 * 24 + 100
    fake = pd.DataFrame({"timestamps": pd.date_range("2026-01-01", periods=n, freq="h"),
                         **{c: np.arange(n, dtype=float) for c in v.COLS}})
    orig = v.fetch_crypto
    v.fetch_crypto = lambda t, i, need: fake.tail(need).reset_index(drop=True)
    try:
        df = v.load_candles("BTCUSDT", "1h", 400, 24, 5, 100)
    finally:
        v.fetch_crypto = orig
    assert len(df) == 400 + 5 * 24
    assert df["close"].iloc[-1] == n - 100 - 1            # offset drops the most recent candles
    ends = []
    for i in range(5):
        ctx = df.iloc[i * 24:i * 24 + 400]
        truth = df.iloc[i * 24 + 400:i * 24 + 424]
        assert ctx["timestamps"].iloc[-1] < truth["timestamps"].iloc[0]
        ends.append((truth["timestamps"].iloc[0], truth["timestamps"].iloc[-1]))
    assert all(ends[i][1] < ends[i + 1][0] for i in range(4))   # real paths do not overlap
    assert df.iloc[4 * 24 + 400:]["timestamps"].iloc[-1] == df["timestamps"].iloc[-1]


def test_too_few_candles():
    orig = v.fetch_crypto
    v.fetch_crypto = lambda t, i, need: pd.DataFrame({"timestamps": [], **{c: [] for c in v.COLS}})
    try:
        v.load_candles("BTCUSDT", "1h", 400, 24, 20, 0)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "Solo hay 0 velas y se necesitan 880" in str(e)
    finally:
        v.fetch_crypto = orig


def test_helpers():
    assert v.is_crypto("BTCUSDT") and v.is_crypto("btc/usdt") and not v.is_crypto("KO")
    a = v.parse_args(["KO", "TLT", "--windows", "5"])
    assert a.tickers == ["KO", "TLT"] and a.lookback == 400 and a.pred_len == 24 and a.windows == 5
    try:
        v.parse_args(["KO", "--windows", "1"])
        raise AssertionError("expected SystemExit")
    except SystemExit:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
