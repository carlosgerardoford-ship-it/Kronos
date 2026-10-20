"""Probabilistic walk-forward validation of Kronos-small as a scenario generator.

Kronos draws S individual trajectories per window (not averaged); they are scored as a predictive distribution
of cumulative log-returns against four statistical baselines that use only the context. Criteria are fixed in
outputs/validation_prob/PREREGISTRO.md. Extra dependencies: pip install -r requirements_prob.txt

usage: python validate_prob.py                       # BTCUSDT 1h, 400 context, 24 ahead, 100 windows, 25 samples
       python validate_prob.py --summarize-only      # recompute summary/plots from the checkpoint (no model)
The run checkpoints every window and resumes automatically if interrupted.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(ROOT_DIR)
import validate as v

MODELS = ["kronos", "normal", "student", "bootstrap", "garch"]
BASELINES = MODELS[1:]
LEVELS = (50, 80, 90)
HORIZONS = (1, 6, 12, 24)
EVENTS = ["touch_up_1", "touch_up_2", "touch_up_3", "touch_dn_1", "touch_dn_2", "touch_dn_3", "dd_2"]
EVENT_LABEL = {"touch_up_1": "toca +1%", "touch_up_2": "toca +2%", "touch_up_3": "toca +3%", "touch_dn_1": "toca -1%",
               "touch_dn_2": "toca -2%", "touch_dn_3": "toca -3%", "dd_2": "drawdown > 2%"}
DM_BANDWIDTH = 4


# ---------------------------------------------------------------- scoring rules

def crps_fair(x, y):
    """Fair (finite-sample corrected) empirical CRPS. x: (m,) or (m, k) samples along axis 0; y: scalar or (k,)."""
    x = np.sort(np.asarray(x, dtype=float), axis=0)
    m = x.shape[0]
    t1 = np.abs(x - y).mean(axis=0)
    w = (2 * np.arange(m) - m + 1).reshape((m,) + (1,) * (x.ndim - 1))
    t2 = (w * x).sum(axis=0) / (m * (m - 1))  # = sum_ij |xi - xj| / (2 m (m - 1))
    return t1 - t2


def energy_score_fair(paths, y):
    """Fair energy score of m trajectories (m, H) against the realised path y (H,)."""
    m = paths.shape[0]
    t1 = np.linalg.norm(paths - y, axis=1).mean()
    t2 = cdist(paths, paths).sum() / (2 * m * (m - 1))
    return t1 - t2


def pit_value(x, y, u):
    """Randomised PIT from samples: (#{x < y} + u) / (m + 1), u ~ U(0,1)."""
    return (np.sum(x < y) + u) / (len(x) + 1)


def central_interval(x, level):
    a = (1 - level / 100) / 2
    # Weibull plotting position, q*(m+1): exact coverage under exchangeability (numpy's default "linear" undercovers with few samples)
    lo, hi = np.quantile(x, [a, 1 - a], method="weibull")
    return float(lo), float(hi)


def var_es(x, alpha):
    var = float(np.quantile(x, alpha, method="weibull"))
    return var, float(x[x <= var].mean())


# ---------------------------------------------------------------- tests of calibration / skill

def _xlogy(k, q):
    return k * np.log(q) if k > 0 else 0.0


def kupiec_pof(x, n, p):
    """Kupiec proportion-of-failures LR test: x failures out of n, expected failure probability p."""
    if n == 0:
        return float("nan"), float("nan")
    pi = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_ind(exc):
    """Christoffersen independence LR test on a 0/1 exceedance sequence (consecutive windows)."""
    e = np.asarray(exc, dtype=int)
    if len(e) < 2 or e.sum() == 0:
        return float("nan"), float("nan")
    a, b = e[:-1], e[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n01 + n11, pi) + _xlogy(n00 + n10, 1 - pi)
    ll1 = 0.0
    if n00 + n01 > 0:
        p01 = n01 / (n00 + n01)
        ll1 += _xlogy(n01, p01) + _xlogy(n00, 1 - p01)
    if n10 + n11 > 0:
        p11 = n11 / (n10 + n11)
        ll1 += _xlogy(n11, p11) + _xlogy(n10, 1 - p11)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def newey_west_lrv(d, bandwidth):
    d = np.asarray(d, dtype=float)
    n = len(d)
    u = d - d.mean()
    s = (u * u).sum() / n
    for k in range(1, bandwidth + 1):
        s += 2 * (1 - k / (bandwidth + 1)) * (u[k:] * u[:-k]).sum() / n
    return s


def diebold_mariano(d, bandwidth=DM_BANDWIDTH):
    """One-sided DM test, H1: E[d] > 0 (d = loss_baseline - loss_model). Newey-West variance, normal p-value."""
    d = np.asarray(d, dtype=float)
    lrv = newey_west_lrv(d, bandwidth)
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(lrv / len(d))
    return float(stat), float(stats.norm.sf(stat))


def block_bootstrap_p(d, block=4, reps=10000, seed=0):
    """One-sided circular block-bootstrap p-value for H1: E[d] > 0 (robustness check)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    nb = -(-n // block)
    idx = (rng.integers(0, n, (reps, nb))[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n] % n
    means = (d - d.mean())[idx].mean(axis=1)
    return float((1 + (means >= d.mean()).sum()) / (reps + 1))


# ---------------------------------------------------------------- path events

def path_events(paths):
    """Boolean event arrays for cumulative log-return paths (n, H); real path: pass y[None, :]."""
    price = np.exp(paths)  # relative to the last context close (= 1)
    simple = price - 1
    peak = np.maximum(1.0, np.maximum.accumulate(price, axis=1))
    dd = ((peak - price) / peak).max(axis=1)
    out = {f"touch_up_{k}": simple.max(axis=1) >= k / 100 for k in (1, 2, 3)}
    out.update({f"touch_dn_{k}": simple.min(axis=1) <= -k / 100 for k in (1, 2, 3)})
    out["dd_2"] = dd > 0.02
    return out


# ---------------------------------------------------------------- baselines (context only)

def sim_normal_rw(returns, H, n, rng):
    return np.cumsum(rng.normal(0.0, returns.std(ddof=1), (n, H)), axis=1)


def sim_student_rw(returns, H, n, rng):
    df, _, scale = stats.t.fit(returns, floc=0)
    return np.cumsum(scale * rng.standard_t(float(np.clip(df, 2.1, 100)), (n, H)), axis=1)


def sim_block_bootstrap(returns, H, n, rng, block=24):
    nb = -(-H // block)
    starts = rng.integers(0, len(returns) - block + 1, (n, nb))
    steps = returns[(starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :H]]
    return np.cumsum(steps, axis=1)


def sim_garch(returns, H, n, seed):
    """GARCH(1,1), zero mean, normal innovations. Returns (paths, ok); ok=False if the fit failed."""
    from arch import arch_model
    try:
        res = arch_model(returns * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off", show_warning=False)
        if res.convergence_flag != 0:
            return None, False
        gen = np.random.default_rng(seed)  # arch ignores random_state; the rng callable is what makes it reproducible
        sims = res.forecast(horizon=H, method="simulation", simulations=n, reindex=False,
                            rng=lambda size: gen.standard_normal(size)).simulations.values[-1]
        paths = np.cumsum(sims, axis=1) / 100
        return (paths, True) if np.isfinite(paths).all() else (None, False)
    except Exception:
        return None, False


def baseline_sims(returns, H, n, seed, i, block):
    """The four baselines from the context log-returns only. Returns (dict name -> (n, H), garch_fallback)."""
    rng = lambda k: np.random.default_rng([seed, i, k])
    out = {"normal": sim_normal_rw(returns, H, n, rng(1)), "student": sim_student_rw(returns, H, n, rng(2)),
           "bootstrap": sim_block_bootstrap(returns, H, n, rng(3), block)}
    garch, ok = sim_garch(returns, H, n, seed + i)
    out["garch"] = garch if ok else sim_normal_rw(returns, H, n, rng(4))
    return out, (not ok)


# ---------------------------------------------------------------- per-window computation

def window_row(i, ctx_end, last_close, y, sims, u_rng):
    """All metrics of one window. y: realised cumulative log-returns (H,); sims: model -> (n, H) paths."""
    H = len(y)
    row = {"window": i, "ctx_end": ctx_end, "last_close": last_close, "real_final": float(y[-1])}
    real_ev = path_events(y[None, :])
    for e in EVENTS:
        row[f"real_{e}"] = bool(real_ev[e][0])
    for name, X in sims.items():
        final, yf = X[:, -1], y[-1]
        row[f"{name}_crps_final"] = float(crps_fair(final, yf))
        for h in HORIZONS:
            if h <= H:
                row[f"{name}_crps_h{h}"] = float(crps_fair(X[:, h - 1], y[h - 1]))
        row[f"{name}_energy"] = float(energy_score_fair(X, y))
        row[f"{name}_pit"] = float(pit_value(final, yf, u_rng.uniform()))
        for lv in LEVELS:
            lo, hi = central_interval(final, lv)
            row[f"{name}_cover{lv}"] = bool(lo <= yf <= hi)
            row[f"{name}_width{lv}"] = hi - lo
        for a in (5, 1):
            var, es = var_es(final, a / 100)
            row[f"{name}_var{a}"], row[f"{name}_es{a}"], row[f"{name}_exceed{a}"] = var, es, bool(yf < var)
        row[f"{name}_disp"] = float(final.std(ddof=1))
        ev = path_events(X)
        for e in EVENTS:
            row[f"{name}_p_{e}"] = float(ev[e].mean())
    return row


def compute_window(df, i, L, P, sampler, n_sims, block, seed):
    """One window. Baselines see only the context; the sampler gets the context and the future timestamps."""
    ctx = df.iloc[i * P:i * P + L]
    truth = df.iloc[i * P + L:i * P + L + P]
    last = float(ctx["close"].iloc[-1])
    y = np.log(truth["close"].to_numpy(dtype=float) / last)
    rets = np.diff(np.log(ctx["close"].to_numpy(dtype=float)))
    kron = np.asarray(sampler(ctx, truth["timestamps"], i), dtype=float)
    sims, fallback = baseline_sims(rets, P, n_sims, seed, i, block)
    sims = {"kronos": kron, **sims}
    row = window_row(i, str(ctx["timestamps"].iloc[-1]), last, y, sims, np.random.default_rng([seed, i, 99]))
    row["garch_fallback"] = fallback
    return row, kron, y


def make_kronos_sampler(predictor, S, seed, T, top_p):
    """S individual (not averaged) trajectories as cumulative log-returns: a batch of S copies, sample_count=1."""
    import torch
    from model.kronos import calc_time_stamps

    def sampler(ctx, truth_ts, i):
        x = ctx[v.COLS].to_numpy(dtype=np.float32)
        mean, std = x.mean(0), x.std(0)
        xn = np.clip((x - mean) / (std + 1e-5), -predictor.clip, predictor.clip)
        xs = calc_time_stamps(ctx["timestamps"].reset_index(drop=True)).to_numpy(dtype=np.float32)
        ys = calc_time_stamps(truth_ts.reset_index(drop=True)).to_numpy(dtype=np.float32)
        torch.manual_seed(seed + i)
        out = predictor.generate(np.repeat(xn[None], S, 0), np.repeat(xs[None], S, 0), np.repeat(ys[None], S, 0),
                                 len(truth_ts), T, 0, top_p, 1, False)
        close = np.maximum((out * (std + 1e-5) + mean)[:, :, 3], 1e-9)
        return np.log(close / float(ctx["close"].iloc[-1]))
    return sampler


# ---------------------------------------------------------------- walk-forward with checkpoint / resume

def _atomic(path, writer):
    tmp = path + ".tmp"
    writer(tmp)
    os.replace(tmp, path)


def run_walk_forward(df, meta, sampler, out_dir, log=print):
    """Run (or resume) all windows, checkpointing the CSV and the Kronos trajectories after every window."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{meta['symbol']}_{meta['interval']}"
    csv_path, npz_path = os.path.join(out_dir, f"{tag}_prob.csv"), os.path.join(out_dir, f"samples_{tag}.npz")
    meta_path = os.path.join(out_dir, "run_meta.json")
    rows, kron, real = [], [], []
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        keys = ("symbol", "interval", "lookback", "pred_len", "windows", "samples", "seed", "temperature", "top_p",
                "baseline_sims", "block", "data_first", "data_last")
        if any(old.get(k) != meta.get(k) for k in keys):
            raise ValueError(f"{out_dir} has a checkpoint with different parameters; use another --out")
        if os.path.exists(csv_path) and os.path.exists(npz_path):
            rows = pd.read_csv(csv_path).to_dict("records")
            z = np.load(npz_path)
            kron, real = list(z["kronos"]), list(z["real"])
            if not (len(rows) == len(kron) == len(real)):
                raise ValueError("checkpoint CSV and trajectories are out of sync")
            log(f"Reanudando: {len(rows)} ventanas ya hechas")
    json.dump(meta, open(meta_path, "w"), indent=1)
    L, P = meta["lookback"], meta["pred_len"]
    t0, start = time.time(), len(rows)
    for i in range(start, meta["windows"]):
        row, k, y = compute_window(df, i, L, P, sampler, meta["baseline_sims"], meta["block"], meta["seed"])
        rows.append(row)
        kron.append(k)
        real.append(y)
        _atomic(csv_path, lambda p: pd.DataFrame(rows).to_csv(p, index=False))
        def _save_npz(p):
            with open(p, "wb") as f:
                np.savez_compressed(f, kronos=np.array(kron), real=np.array(real))
        _atomic(npz_path, _save_npz)
        done = i + 1 - start
        eta = (time.time() - t0) / done * (meta["windows"] - i - 1) / 60
        log(f"Ventana {i + 1} de {meta['windows']} · faltan ~{eta:.0f} min")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary

def _table(header, rows):
    cells = [header] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in cells)


def summarize(df, meta):
    n, H = len(df), meta["pred_len"]
    hz = [h for h in HORIZONS if h <= H]
    L = [f"VALIDACION PROBABILISTICA DE KRONOS-SMALL: {meta['symbol']} {meta['interval']}",
         f"Ventanas: {n} (contexto {meta['lookback']}, horizonte {H}) · muestras de Kronos: {meta['samples']} · "
         f"baselines: {meta['baseline_sims']} trayectorias",
         f"Periodo (fin del contexto): {df['ctx_end'].iloc[0]} -> {df['ctx_end'].iloc[-1]}",
         f"Ventanas de GARCH que usaron el respaldo del paseo normal: {int(df['garch_fallback'].sum())} de {n}", ""]

    # (a) skill
    crps = {m: df[f"{m}_crps_final"].mean() for m in MODELS}
    best = min(BASELINES, key=lambda m: crps[m])
    d = (df[f"{best}_crps_final"] - df["kronos_crps_final"]).to_numpy()
    dm_stat, dm_p = diebold_mariano(d)
    bb_p = block_bootstrap_p(d)
    skill = {b: 1 - crps["kronos"] / crps[b] for b in BASELINES}
    a_ok = skill[best] > 0 and dm_p < 0.05
    L += ["== (a) HABILIDAD: CRPS justo del cierre final (log-retorno acumulado a 24 h) ==",
          _table(["modelo", "CRPS medio", "skill de Kronos vs este"],
                 [[m, f"{crps[m]:.6f}", "-" if m == "kronos" else f"{skill[m]:+.3f}"] for m in MODELS]),
          f"Mejor baseline: {best} (CRPS medio {crps[best]:.6f}); skill score = {skill[best]:+.4f}",
          f"Diebold-Mariano (Newey-West, ancho de banda {DM_BANDWIDTH}, una cola): estadistico {dm_stat:+.3f}, p = {dm_p:.4f}",
          f"  Robustez, bootstrap por bloques (4 ventanas, 10000 remuestreos): p = {bb_p:.4f}",
          f"Criterio (a) [skill > 0 y p < 0,05]: {'CUMPLE' if a_ok else 'NO CUMPLE'}", ""]

    # (b) calibration
    cal_rows, b_ok = [], True
    for lv in LEVELS:
        x = int((~df[f"kronos_cover{lv}"].astype(bool)).sum())
        lr, p = kupiec_pof(x, n, 1 - lv / 100)
        b_ok &= p > 0.05
        cal_rows.append([f"{lv}%", f"{(n - x) / n * 100:.0f}%", x, f"{lr:.3f}", f"{p:.4f}",
                         f"{df[f'kronos_width{lv}'].mean():.4f}"])
    ks = stats.kstest(df["kronos_pit"].to_numpy(), "uniform")
    b_ok &= ks.pvalue > 0.05
    L += ["== (b) CALIBRACION del cierre final de Kronos ==",
          _table(["intervalo", "cobertura obs.", "fallas", "Kupiec LR", "Kupiec p", "ancho medio"], cal_rows),
          f"PIT (aleatorizado) vs uniforme, Kolmogorov-Smirnov: D = {ks.statistic:.4f}, p = {ks.pvalue:.4f}",
          f"Criterio (b) [Kupiec p > 0,05 en 50/80/90% y KS p > 0,05]: {'CUMPLE' if b_ok else 'NO CUMPLE'}", ""]
    verdict = "PASA" if (a_ok and b_ok) else "NO PASA"
    L += [f"VEREDICTO CONTRA EL PRERREGISTRO [(a) Y (b)]: {verdict}", "",
          "---------------- METRICAS SECUNDARIAS (no definen la aprobacion) ----------------", ""]

    # secondary: horizons, energy, coverage of all models
    rows = [[f"h{h}", *[f"{1 - df[f'kronos_crps_h{h}'].mean() / df[f'{b}_crps_h{h}'].mean():+.3f}" for b in BASELINES]]
            for h in hz]
    rows.append(["energy", *[f"{1 - df['kronos_energy'].mean() / df[f'{b}_energy'].mean():+.3f}" for b in BASELINES]])
    L += ["Skill score de Kronos por horizonte y energy score (vs cada baseline):",
          _table(["", *BASELINES], rows), ""]
    rows = []
    for m in MODELS:
        rows.append([m, *[f"{df[f'{m}_cover{lv}'].mean() * 100:.0f}% / {df[f'{m}_width{lv}'].mean():.4f}" for lv in LEVELS]])
    L += ["Cobertura observada / ancho medio de los intervalos (por modelo):",
          _table(["modelo", *[f"{lv}%" for lv in LEVELS]], rows), ""]

    # events
    rows, extreme = [], []
    for e in EVENTS:
        o = df[f"real_{e}"].astype(float).to_numpy()
        rows.append([EVENT_LABEL[e], f"{o.mean():.2f}", *[f"{((df[f'{m}_p_{e}'] - o) ** 2).mean():.4f}" for m in MODELS]])
        extreme.append([EVENT_LABEL[e], *[int(((df[f'{m}_p_{e}'] == 0) | (df[f'{m}_p_{e}'] == 1)).sum()) for m in MODELS]])
    L += ["Eventos de camino, Brier score (sin recortar probabilidades; menor es mejor):",
          _table(["evento", "frec. real", *MODELS], rows),
          "Ventanas con probabilidad predicha exactamente 0 o 1 (por falta de muestras), de " + str(n) + ":",
          _table(["evento", *MODELS], extreme), ""]

    # tails
    rows = []
    for a in (5, 1):
        for m in MODELS:
            exc = df[f"{m}_exceed{a}"].astype(int).to_numpy()
            _, pk = kupiec_pof(int(exc.sum()), n, a / 100)
            _, pc = christoffersen_ind(exc)
            es_real = df.loc[exc == 1, "real_final"].mean() if exc.sum() else float("nan")
            flag = "NO CONFIABLE, excluido" if (a == 1 and m == "kronos") else ""
            rows.append([f"{a}%", m, int(exc.sum()), f"{a / 100 * n:.1f}", f"{pk:.3f}", f"{pc:.3f}",
                         f"{df[f'{m}_es{a}'].mean():+.4f}", f"{es_real:+.4f}", flag])
    L += ["Colas del cierre final (VaR/ES de la cola izquierda; excedencia = real < VaR):",
          _table(["alfa", "modelo", "excedencias", "esperadas", "Kupiec p", "Christoff. ind. p", "ES medio pred.",
                  "retorno real en exced.", "nota"], rows),
          "  (El VaR/ES al 1% de Kronos, con pocas muestras, queda fuera del rango muestral: no se usa en ninguna conclusion.)", ""]

    # regime
    absr = df["real_final"].abs()
    rk, rg = stats.spearmanr(df["kronos_disp"], absr)[0], stats.spearmanr(df["garch_disp"], absr)[0]
    L += ["Regimen: Spearman entre la dispersion predicha y |retorno final| realizado:",
          f"  Kronos {rk:+.3f} · GARCH {rg:+.3f}"]
    return "\n".join(L), verdict


def make_plots(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3 * len(MODELS), 3), sharey=True)
    for ax, m in zip(axes, MODELS):
        ax.hist(df[f"{m}_pit"], bins=10, range=(0, 1), color="#3b82f6")
        ax.axhline(len(df) / 10, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"PIT {m}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pit_hist.png"), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.15
    for j, m in enumerate(MODELS):
        ax.bar(np.arange(len(LEVELS)) + j * width, [df[f"{m}_cover{lv}"].mean() * 100 for lv in LEVELS], width, label=m)
    for k, lv in enumerate(LEVELS):
        ax.hlines(lv, k - 0.1, k + 0.8, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(LEVELS)) + 2 * width)
    ax.set_xticklabels([f"{lv}%" for lv in LEVELS])
    ax.set_ylabel("cobertura observada (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coverage.png"), dpi=130)
    plt.close(fig)


def write_summary(df, meta, out_dir):
    text, verdict = summarize(df, meta)
    with open(os.path.join(out_dir, "summary_prob.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    make_plots(df, out_dir)
    return text, verdict


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--windows", type=int, default=100)
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline-sims", type=int, default=1000)
    p.add_argument("--block", type=int, default=24)
    p.add_argument("--out", default=os.path.join(ROOT_DIR, "outputs", "validation_prob"))
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args(argv)
    if a.samples < 2 or a.windows < 5 or a.pred_len < 1 or a.block < 1 or a.block >= a.lookback:
        p.error("invalid --samples/--windows/--pred-len/--block")
    return a


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    input_csv = os.path.join(a.out, f"{a.symbol}_{a.interval}_input.csv")
    if os.path.exists(input_csv):  # pinned candles so a resumed run sees exactly the same data
        df = pd.read_csv(input_csv, parse_dates=["timestamps"])
    elif a.summarize_only:
        raise SystemExit("no checkpoint found in --out")
    else:
        df = v.load_candles(a.symbol, a.interval, a.lookback, a.pred_len, a.windows, 0)
        df.to_csv(input_csv, index=False)
    meta = {"symbol": a.symbol, "interval": a.interval, "lookback": a.lookback, "pred_len": a.pred_len,
            "windows": a.windows, "samples": a.samples, "seed": a.seed, "temperature": a.temperature, "top_p": a.top_p,
            "baseline_sims": a.baseline_sims, "block": a.block,
            "data_first": str(df["timestamps"].iloc[0]), "data_last": str(df["timestamps"].iloc[-1])}
    if a.summarize_only:
        rows = pd.read_csv(os.path.join(a.out, f"{a.symbol}_{a.interval}_prob.csv"))
        text, _ = write_summary(rows, json.load(open(os.path.join(a.out, "run_meta.json"))), a.out)
    else:
        sampler = make_kronos_sampler(v.get_predictor(), a.samples, a.seed, a.temperature, a.top_p)
        rows = run_walk_forward(df, meta, sampler, a.out, log=lambda s: print(s, flush=True))
        text, _ = write_summary(rows, meta, a.out)
    print(text)


if __name__ == "__main__":
    main()
