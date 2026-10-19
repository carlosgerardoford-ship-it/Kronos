"""Lag of the prediction vs the real path from validate.py CSVs, by correlation of candle-to-candle changes.

usage: python analyze_lag.py outputs/validation [pred_len]
Compares correct (prediction i vs real i) with unrelated (prediction i vs real j != i) pairs, each re-anchored
to its own last close. If both look alike, the measured lag carries no information.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import validate as v

folder = sys.argv[1] if len(sys.argv) > 1 else "../outputs/validation"
files = sorted(f for f in glob.glob(os.path.join(folder, "*.csv")))
lines = []
for f in files:
    d = pd.read_csv(f)
    if "close_pred" not in d:
        continue
    pred = [np.array([float(x) for x in s.split(";")]) for s in d.close_pred]
    real = [np.array([float(x) for x in s.split(";")]) for s in d.close_real]
    last = d.last_close.to_numpy()
    n, P = len(d), len(pred[0])
    ms = int(sys.argv[2]) // 4 if len(sys.argv) > 2 else max(1, P // 4)
    ok = [v.best_lag_corr(pred[i], real[i], last[i], ms) for i in range(n)]
    un = [v.best_lag_corr(pred[i] - last[i], real[j] - last[j], 0.0, ms) for i in range(n) for j in range(n) if i != j]
    mk, nk = np.array([x[0] for x in ok]), np.array([x[0] for x in un])
    c0m, c0n = np.nanmean([x[2] for x in ok]), np.nanmean([x[2] for x in un])
    edge = lambda k: np.mean(np.abs(k) == ms) * 100
    name = os.path.basename(f).split("_")[0]
    lines.append(f"{name:8s} corr@0 {c0m:+.3f} vs {c0n:+.3f} | desfase abs {np.abs(mk).mean():.1f} vs {np.abs(nk).mean():.1f} | "
                 f"adelantado {np.mean(mk < 0) * 100:.0f}% vs {np.mean(nk < 0) * 100:.0f}% | borde {edge(mk):.0f}% vs {edge(nk):.0f}%")
head = f"Desfase por correlación de cambios (±{ms} velas). Formato: pares correctos vs pares sin relación"
out = "\n".join([head] + lines)
print(out)
with open(os.path.join(folder, "lag_corrected.txt"), "w", encoding="utf-8") as fh:
    fh.write(out + "\n")
