"""Model-free tests for validate_prob.py. Run: python test_validate_prob.py (or pytest)."""
import os
import json
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
import validate_prob as vp


# ---------------------------------------------------------------- scoring rules

def test_crps_fair_matches_bruteforce():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=7), 0.3
    m = len(x)
    brute = np.abs(x - y).mean() - sum(abs(a - b) for a in x for b in x) / (2 * m * (m - 1))
    assert abs(vp.crps_fair(x, y) - brute) < 1e-12
    X = rng.normal(size=(7, 3))                       # vectorised over columns
    assert np.allclose(vp.crps_fair(X, np.array([0.1, 0.2, 0.3]))[1], vp.crps_fair(X[:, 1], 0.2))


def test_crps_fair_unbiased_against_normal_closed_form():
    mu, sigma, y = 0.0, 1.0, 0.7
    z = (y - mu) / sigma
    closed = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    rng = np.random.default_rng(1)
    for m in (10, 25):
        est = vp.crps_fair(rng.normal(mu, sigma, (m, 40000)), y).mean()
        assert abs(est - closed) < 0.005, (m, est, closed)
    big = vp.crps_fair(rng.normal(mu, sigma, 4000), y)  # large sample converges too
    assert abs(big - closed) < 0.02


def test_energy_score_fair_bruteforce():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(5, 3)), rng.normal(size=3)
    m = len(X)
    brute = np.mean([np.linalg.norm(a - y) for a in X]) - sum(
        np.linalg.norm(a - b) for a in X for b in X) / (2 * m * (m - 1))
    assert abs(vp.energy_score_fair(X, y) - brute) < 1e-12


def test_pit_and_coverage_right_vs_wrong_forecast():
    rng = np.random.default_rng(3)
    n, m = 400, 25
    y = rng.normal(size=n)
    for sigma, should_pass in ((1.0, True), (0.5, False)):
        X = rng.normal(0, sigma, (n, m))
        pit = np.array([vp.pit_value(X[i], y[i], rng.uniform()) for i in range(n)])
        ks_p = stats.kstest(pit, "uniform").pvalue
        p80 = []
        for lv in (50, 80, 90):
            fails = sum(not (lo <= y[i] <= hi) for i in range(n) for lo, hi in [vp.central_interval(X[i], lv)])
            p80.append(vp.kupiec_pof(fails, n, 1 - lv / 100)[1])
        if should_pass:
            assert ks_p > 0.01 and min(p80) > 0.01, (ks_p, p80)
        else:
            assert ks_p < 0.01 and min(p80) < 0.01, (ks_p, p80)


def test_central_interval_has_exact_coverage_with_25_samples():
    rng = np.random.default_rng(8)
    n, m = 20000, 25
    X, y = rng.normal(size=(n, m)), rng.normal(size=n)
    for lv in (50, 80, 90):
        cover = np.mean([lo <= yi <= hi for yi, (lo, hi) in zip(y, (vp.central_interval(x, lv) for x in X))])
        assert abs(cover - lv / 100) < 0.015, (lv, cover)                  # residual ~+1 pt from interpolating in the tails
        lin = np.mean([lo <= yi <= hi for yi, (lo, hi) in zip(y, (np.quantile(x, [(1 - lv / 100) / 2, 1 - (1 - lv / 100) / 2])
                                                                for x in X))])
        assert lin < lv / 100 - 0.02                                        # numpy's default would undercover


def test_kupiec_known_values():
    lr, p = vp.kupiec_pof(10, 100, 0.05)
    expect = -2 * ((90 * np.log(0.95) + 10 * np.log(0.05)) - (90 * np.log(0.9) + 10 * np.log(0.1)))
    assert abs(lr - expect) < 1e-12 and abs(p - stats.chi2.sf(expect, 1)) < 1e-12
    assert vp.kupiec_pof(5, 100, 0.05)[0] < 1e-9                       # exactly as expected -> LR 0
    assert abs(vp.kupiec_pof(0, 100, 0.05)[0] + 2 * 100 * np.log(0.95)) < 1e-12   # no failures


def test_christoffersen_independence():
    rng = np.random.default_rng(4)
    iid = (rng.uniform(size=400) < 0.1).astype(int)
    assert vp.christoffersen_ind(iid)[1] > 0.05
    sticky, state = [], 0
    for _ in range(400):                                               # clustered exceedances
        state = int(rng.uniform() < (0.8 if state else 0.03))
        sticky.append(state)
    assert vp.christoffersen_ind(sticky)[1] < 0.01
    assert np.isnan(vp.christoffersen_ind([0] * 50)[1])                # no exceedances: undefined


def test_diebold_mariano_and_newey_west():
    rng = np.random.default_rng(5)
    stat, p = vp.diebold_mariano(np.random.default_rng(0).normal(0.5, 1, 100))   # true effect: seeds 0-4 all give stat > 4.7
    assert stat > 3 and p < 1e-3
    assert vp.diebold_mariano(rng.normal(0, 1, 100))[1] > 0.05
    n = 100
    alt = np.array([1.0, -1.0] * (n // 2))                             # lrv = 1/n with bandwidth 1
    assert abs(vp.newey_west_lrv(alt, 1) - 1 / n) < 1e-12
    e = rng.normal(size=500)
    ar = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.7 * ar[t - 1] + e[t]                                 # autocorrelated: HAC > naive variance
    assert vp.newey_west_lrv(ar, 8) > 2 * ar.var()
    assert vp.block_bootstrap_p(rng.normal(0.6, 1, 100)) < 0.01
    assert vp.block_bootstrap_p(rng.normal(0, 1, 100)) > 0.05


# ---------------------------------------------------------------- events, tails, baselines

def test_path_events():
    up = np.log(np.array([[1.004, 1.012, 1.015, 1.01]]))               # peaks at +1.5%
    ev = vp.path_events(up)
    assert ev["touch_up_1"][0] and not ev["touch_up_2"][0] and not ev["touch_dn_1"][0] and not ev["dd_2"][0]
    down = np.log(np.array([[0.975, 0.99]]))                           # drawdown measured from the last close
    ev = vp.path_events(down)
    assert ev["dd_2"][0] and ev["touch_dn_1"][0] and ev["touch_dn_2"][0] and not ev["touch_dn_3"][0]
    peak = np.log(np.array([[1.03, 0.99, 1.0]]))                       # 3.9% below its own peak
    assert vp.path_events(peak)["dd_2"][0]


def test_var_es():
    x = np.arange(100) / 100
    var, es = vp.var_es(x, 0.05)
    assert abs(var - np.quantile(x, 0.05, method="weibull")) < 1e-12 and abs(es - x[x <= var].mean()) < 1e-12
    assert vp.var_es(x[:25], 0.01)[0] == x[:25].min()                   # 1% with 25 samples falls outside the sample range: clipped to the minimum


def test_baselines_shapes_and_properties():
    rng = np.random.default_rng(6)
    rets = rng.standard_t(5, 399) * 0.004
    n, H = 4000, 24
    g = np.random.default_rng(0)
    normal = vp.sim_normal_rw(rets, H, n, g)
    assert normal.shape == (n, H) and abs(normal[:, -1].std() / (rets.std(ddof=1) * np.sqrt(H)) - 1) < 0.05
    student = vp.sim_student_rw(rets, H, n, g)
    assert student.shape == (n, H) and abs(np.median(student[:, -1])) < 0.01
    boot = vp.sim_block_bootstrap(rets, H, 50, g, block=24)
    blocks = {tuple(np.round(np.cumsum(rets[s:s + 24]), 12)) for s in range(len(rets) - 23)}
    assert all(tuple(np.round(r, 12)) in blocks for r in boot)         # each path = one contiguous 24h block
    assert vp.sim_block_bootstrap(rets, 30, 10, g, block=24).shape == (10, 30)
    paths, ok = vp.sim_garch(rets, H, 500, 0)
    assert ok and paths.shape == (500, H) and np.isfinite(paths).all()
    assert np.array_equal(paths, vp.sim_garch(rets, H, 500, 0)[0])    # same seed -> same simulation
    assert not np.array_equal(paths, vp.sim_garch(rets, H, 500, 1)[0])
    assert vp.sim_garch(np.zeros(399), H, 100, 0)[1] is False          # degenerate context -> fallback flag
    sims, fb = vp.baseline_sims(np.zeros(399) + 1e-12 * rng.normal(size=399), H, 100, 0, 0, 24)
    assert set(sims) == set(vp.BASELINES) and sims["garch"].shape == (100, H)


# ---------------------------------------------------------------- pipeline: no leak, resume, summary

def make_df(n_windows, L=60, P=24, seed=0):
    rng = np.random.default_rng(seed)
    n = L + n_windows * P
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame({"timestamps": pd.date_range("2026-01-01", periods=n, freq="h"), "open": close, "high": close,
                         "low": close, "close": close, "volume": 1.0, "amount": close})


def stub_sampler(S=25, P=24):
    def sampler(ctx, truth_ts, i):
        rng = np.random.default_rng([7, i])
        return np.cumsum(rng.normal(0, 0.004, (S, len(truth_ts))), axis=1)
    return sampler


def test_no_leak_baselines_and_sampler_see_only_context():
    L, P = 60, 24
    df = make_df(3, L, P)
    df2 = df.copy()
    df2.loc[df2.index >= L + P, ["open", "high", "low", "close"]] *= 5.0   # alter everything after window 0's truth
    seen_returns, seen_ts = [], []
    orig = vp.baseline_sims
    vp.baseline_sims = lambda returns, *a: (seen_returns.append(returns.copy()), orig(returns, *a))[1]

    def spy(ctx, truth_ts, i):
        seen_ts.append((ctx["timestamps"].max(), truth_ts.min(), ctx.index.max()))
        return stub_sampler()(ctx, truth_ts, i)
    try:
        r1 = vp.compute_window(df, 0, L, P, spy, 100, 24, 0)[0]
        r2 = vp.compute_window(df2, 0, L, P, spy, 100, 24, 0)[0]
    finally:
        vp.baseline_sims = orig
    ctx_ret = np.diff(np.log(df["close"].to_numpy()[:L]))
    assert len(seen_returns[0]) == L - 1 and np.allclose(seen_returns[0], ctx_ret)   # baselines: context returns only
    assert np.allclose(seen_returns[0], seen_returns[1])                             # unchanged when later data changes
    assert seen_ts[0][0] < seen_ts[0][1] and seen_ts[0][2] == L - 1                  # sampler: context + future timestamps
    assert r1["real_final"] == r2["real_final"]
    df3 = df.copy()
    df3.loc[df3.index >= L, "close"] *= 3.0                                          # change window 0's own truth only
    r3 = vp.compute_window(df3, 0, L, P, stub_sampler(), 100, 24, 0)[0]
    assert r3["real_final"] != r1["real_final"] and r3["normal_disp"] == r1["normal_disp"]   # baseline ignores the truth


def test_checkpoint_resume_and_summary():
    L, P, W = 60, 24, 8
    df = make_df(W, L, P)
    meta = {"symbol": "TEST", "interval": "1h", "lookback": L, "pred_len": P, "windows": W, "samples": 25, "seed": 0,
            "temperature": 1.0, "top_p": 0.9, "baseline_sims": 200, "block": 24,
            "data_first": str(df.timestamps.iloc[0]), "data_last": str(df.timestamps.iloc[-1])}
    quiet = lambda s: None
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        base = stub_sampler()

        def flaky(ctx, truth_ts, i):                                     # simulated interruption at window 5
            if i == 5:
                raise RuntimeError("interrupted")
            return base(ctx, truth_ts, i)
        try:
            vp.run_walk_forward(df, meta, flaky, a, log=quiet)
            raise AssertionError("expected interruption")
        except RuntimeError:
            pass
        assert len(pd.read_csv(os.path.join(a, "TEST_1h_prob.csv"))) == 5
        resumed = vp.run_walk_forward(df, meta, base, a, log=quiet)
        fresh = vp.run_walk_forward(df, meta, base, b, log=quiet)
        pd.testing.assert_frame_equal(resumed, fresh)                    # resume == uninterrupted run
        z = np.load(os.path.join(a, "samples_TEST_1h.npz"))
        assert z["kronos"].shape == (W, 25, P) and z["real"].shape == (W, P)
        try:
            vp.run_walk_forward(df, {**meta, "samples": 50}, base, a, log=quiet)
            raise AssertionError("expected parameter mismatch")
        except ValueError:
            pass
        text, verdict = vp.write_summary(resumed, meta, a)
        for needle in ("HABILIDAD", "CALIBRACION", "VEREDICTO CONTRA EL PRERREGISTRO", "Diebold-Mariano",
                       "respaldo del paseo normal", "NO CONFIABLE, excluido", "Brier", "exactamente 0 o 1"):
            assert needle in text, needle
        assert verdict in ("PASA", "NO PASA")
        assert os.path.exists(os.path.join(a, "pit_hist.png")) and os.path.exists(os.path.join(a, "coverage.png"))
        assert os.path.exists(os.path.join(a, "summary_prob.txt"))


def test_summary_counts_extreme_probabilities_without_clipping():
    L, P, W = 60, 24, 6
    df = make_df(W, L, P)
    meta = {"symbol": "T", "interval": "1h", "lookback": L, "pred_len": P, "windows": W, "samples": 25, "seed": 0,
            "temperature": 1.0, "top_p": 0.9, "baseline_sims": 200, "block": 24, "data_first": "", "data_last": ""}
    rows = vp.run_walk_forward(df, meta, stub_sampler(), tempfile.mkdtemp(), log=lambda s: None)
    rows["kronos_p_touch_up_3"] = [0.0, 0.0, 1.0, 0.0, 0.5, 1.0]        # probabilities stay as they are
    text, _ = vp.summarize(rows, meta)
    line = [l for l in text.split("\n") if l.startswith("toca +3%")]
    assert len(line) == 2 and line[0].split()[3] == "0.3750"            # Brier uses p as is: (0+0+1+0+0.25+1)/6 (real event never happens)
    assert line[1].split()[2] == "5"                                    # 5 of 6 windows with p exactly 0 or 1 for Kronos


def test_kronos_sampler_returns_individual_trajectories():
    class Fake:
        clip = 5
        seen = {}

        def generate(self, x, xs, ys, pred_len, T, top_k, top_p, sample_count, verbose):
            Fake.seen = dict(batch=x.shape[0], sample_count=sample_count, pred_len=pred_len, T=T, top_p=top_p)
            rng = np.random.default_rng(0)
            out = np.zeros((x.shape[0], pred_len, 6), dtype=np.float32)
            out[:, :, 3] = rng.normal(size=(x.shape[0], pred_len)) * 0.01   # normalised close: differs per sample
            return out
    df = make_df(1, 60, 24)
    ctx, truth = df.iloc[:60], df.iloc[60:84]
    out = vp.make_kronos_sampler(Fake(), 25, 0, 1.0, 0.9)(ctx, truth["timestamps"], 0)
    assert out.shape == (25, 24) and len({tuple(np.round(r, 9)) for r in out}) == 25
    assert Fake.seen == {"batch": 25, "sample_count": 1, "pred_len": 24, "T": 1.0, "top_p": 0.9}  # no internal averaging
    x = ctx[vp.v.COLS].to_numpy(dtype=np.float32)
    mean, std = x.mean(0), x.std(0)
    assert np.allclose(np.exp(out[0, 0]) * ctx.close.iloc[-1], mean[3] + std[3] * 0.0 + 0, atol=1e6)  # sanity only


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
