import os
import sys
import json
import queue
import threading
import urllib.error
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(ROOT_DIR)
from model import Kronos, KronosTokenizer, KronosPredictor

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
TZ = "America/New_York"
COLS = ["open", "high", "low", "close", "volume", "amount"]
CRYPTO_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD")
INTERVALS = ["5m", "15m", "1h", "4h", "1d"]
YAHOO_RANGE = {"5m": "60d", "15m": "60d", "1h": "1y", "1d": "5y"}

BG = "#1e1e1e"
PANEL = "#252526"
FIELD = "#2d2d30"
FG = "#e6e6e6"
MUTED = "#9da0a6"
ACCENT = "#3b82f6"
HISTORY = "#60a5fa"
PREDICTION = "#f87171"
TRUTH = "#e6e6e6"


# ---------------------------------------------------------------- data

def detect_kind(symbol):
    return "crypto" if "/" in symbol or symbol.upper().endswith(CRYPTO_QUOTES) else "stock"


def fetch_crypto(symbol, interval, limit):
    # Binance spot candles (UTC); the still-open candle is dropped
    pair = symbol.upper().replace("/", "").replace("-", "").replace(" ", "")
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={pair}&interval={interval}&limit={limit + 1}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            rows = json.load(r)[:-1]
    except urllib.error.HTTPError as e:
        raise ValueError(f"Binance no devolvió datos para {pair} (HTTP {e.code}). Revisá el par, por ejemplo BTCUSDT.")
    df = pd.DataFrame(rows).iloc[:, :8]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "amount"]
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in COLS:
        df[c] = df[c].astype(float)
    return df[["timestamps"] + COLS].reset_index(drop=True)


def fetch_stock(symbol, interval, limit):
    # Yahoo Finance candles in New York local time; the still-open candle is dropped
    if interval not in YAHOO_RANGE:
        raise ValueError(f"Yahoo Finance no ofrece velas de {interval} para acciones. Usá 5m, 15m, 1h o 1d.")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
           f"?interval={interval}&range={YAHOO_RANGE[interval]}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.load(r)["chart"]["result"][0]
    except urllib.error.HTTPError as e:
        raise ValueError(f"Yahoo Finance no devolvió datos para {symbol.upper()} (HTTP {e.code}). Revisá el ticker.")
    q = res["indicators"]["quote"][0]
    start = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(TZ)
    df = pd.DataFrame({"start": start, **{c: q[c] for c in ["open", "high", "low", "close", "volume"]}}).dropna()
    df = df[df["volume"] > 0]  # Yahoo adds a zero-volume placeholder candle for the last session
    if interval == "1d":
        end = df["start"].dt.normalize() + pd.Timedelta(hours=16)
    else:
        end = df["start"] + pd.Timedelta(interval)
    df = df[end <= pd.Timestamp.now(tz=TZ)]
    df["timestamps"] = df["start"].dt.tz_localize(None)
    if interval == "1d":
        df["timestamps"] = df["timestamps"].dt.normalize()
    df["amount"] = df["close"] * df["volume"]  # Yahoo has no turnover: approximated
    return df[["timestamps"] + COLS].tail(limit).reset_index(drop=True)


def future_timestamps(kind, last, interval, n):
    step = pd.Timedelta(interval)
    if kind == "crypto":
        return pd.Series(pd.date_range(last + step, periods=n, freq=step))
    if interval == "1d":
        return pd.Series(pd.bdate_range(last.normalize() + pd.Timedelta(days=1), periods=n))
    # Regular session 09:30-16:00 ET, Mon-Fri (holidays not excluded)
    per_day = -(-390 // int(step.total_seconds() // 60))
    days = pd.bdate_range(last.normalize(), periods=n // per_day + 3)
    slots = [d + pd.Timedelta(hours=9, minutes=30) + k * step for d in days for k in range(per_day)]
    return pd.Series([t for t in slots if t > last][:n])


# ---------------------------------------------------------------- model

_predictor = None


def get_predictor(status):
    global _predictor
    if _predictor is None:
        status("Cargando modelo (la primera vez descarga los pesos)...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        _predictor = KronosPredictor(model, tokenizer, max_context=512)
    return _predictor


def run_analysis(symbol, kind, interval, lookback, pred_len, samples, backtest, status):
    if kind == "auto":
        kind = detect_kind(symbol)
    status(f"Descargando velas de {symbol.upper()} ({'Binance' if kind == 'crypto' else 'Yahoo Finance'})...")
    need = lookback + pred_len if backtest else lookback
    df = (fetch_crypto if kind == "crypto" else fetch_stock)(symbol, interval, need)
    if len(df) < need:
        raise ValueError(f"Solo hay {len(df)} velas disponibles y se necesitan {need}. Bajá el lookback o la cantidad a predecir.")
    predictor = get_predictor(status)
    x_df = df.loc[:lookback - 1, COLS]
    x_ts = df.loc[:lookback - 1, "timestamps"]
    truth = None
    if backtest:
        truth = df.loc[lookback:lookback + pred_len - 1].reset_index(drop=True)
        y_ts = truth["timestamps"]
    else:
        y_ts = future_timestamps(kind, x_ts.iloc[-1], interval, pred_len)
    status(f"Generando predicción ({pred_len} velas, {samples} muestra(s))...")
    pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len,
                             T=1.0, top_p=0.9, sample_count=samples, verbose=False)
    return {"symbol": symbol.upper().replace("/", ""), "kind": kind, "interval": interval, "backtest": backtest,
            "context": df.loc[:lookback - 1].reset_index(drop=True), "truth": truth, "pred": pred}


# ---------------------------------------------------------------- summary

def summarize(r):
    ctx, pred, truth = r["context"], r["pred"], r["truth"]
    last = ctx["close"].iloc[-1]
    end = pred["close"].iloc[-1]
    fmt = lambda t: t.strftime("%Y-%m-%d %H:%M")
    mode = "Backtest (velas ya ocurridas)" if r["backtest"] else "Pronóstico"
    lines = [
        f"{r['symbol']}  ·  {r['interval']}  ·  {mode}",
        f"Fuente: {'Binance' if r['kind'] == 'crypto' else 'Yahoo Finance'}",
        "",
        f"Contexto: {fmt(ctx['timestamps'].iloc[0])}  →  {fmt(ctx['timestamps'].iloc[-1])}",
        f"Predicción: {fmt(pred.index[0])}  →  {fmt(pred.index[-1])}",
        "",
        f"Último cierre:        {last:,.2f}",
        f"Cierre final pred.:   {end:,.2f}   ({(end / last - 1) * 100:+.2f}%)",
        f"Máximo pred.:         {pred['close'].max():,.2f}   ({fmt(pred['close'].idxmax())})",
        f"Mínimo pred.:         {pred['close'].min():,.2f}   ({fmt(pred['close'].idxmin())})",
        f"Tendencia pred.:      {'alcista' if end > last else 'bajista'}",
    ]
    if truth is not None:
        real_end = truth["close"].iloc[-1]
        mae = (truth["close"].values - pred["close"].values).__abs__().mean()
        hit = (real_end > last) == (end > last)
        lines += ["", f"Cierre final real:    {real_end:,.2f}   ({(real_end / last - 1) * 100:+.2f}%)",
                  f"Error abs. medio:     {mae:,.2f}",
                  f"Dirección acertada:   {'sí' if hit else 'no'}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- GUI

class App:
    def __init__(self, root):
        self.root = root
        self.result = None
        self.q = queue.Queue()
        root.title("Kronos · Pronóstico de velas")
        root.geometry("1200x760")
        root.minsize(980, 620)
        root.configure(bg=BG)
        self._dark_titlebar()
        self._style()
        self._build()
        self.root.after(100, self._poll)

    def _dark_titlebar(self):
        if sys.platform != "win32":
            return
        try:
            import ctypes
            self.root.update()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
        except Exception:
            pass

    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=FIELD, bordercolor="#3c3c3c",
                    lightcolor=BG, darkcolor=BG, troughcolor=FIELD, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=MUTED)
        s.configure("TEntry", fieldbackground=FIELD, foreground=FG, insertcolor=FG)
        s.configure("TSpinbox", fieldbackground=FIELD, foreground=FG, arrowcolor=FG, insertcolor=FG)
        s.configure("TCombobox", fieldbackground=FIELD, foreground=FG, arrowcolor=FG, background=FIELD)
        s.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", FG)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", FG)])
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("TButton", background="#3a3a3d", foreground=FG, padding=(12, 6), borderwidth=0)
        s.map("TButton", background=[("active", "#4a4a4d"), ("disabled", "#2a2a2c")],
              foreground=[("disabled", "#6b6b6b")])
        s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", padding=(16, 6))
        s.map("Accent.TButton", background=[("active", "#5b9bff"), ("disabled", "#2b4a80")],
              foreground=[("disabled", "#9db7e6")])
        s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=FIELD, borderwidth=0)
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

    def _field(self, parent, label, widget, col):
        ttk.Label(parent, text=label).grid(row=0, column=col, sticky="w", padx=(0, 10))
        widget.grid(row=1, column=col, sticky="we", padx=(0, 10), pady=(2, 0))

    def _build(self):
        bar = ttk.Frame(self.root, padding=(14, 12, 4, 8))
        bar.pack(fill="x")
        self.symbol = tk.StringVar(value="BTCUSDT")
        self.kind = tk.StringVar(value="Auto")
        self.interval = tk.StringVar(value="1h")
        self.lookback = tk.IntVar(value=400)
        self.pred_len = tk.IntVar(value=24)
        self.samples = tk.IntVar(value=1)
        self.backtest = tk.BooleanVar(value=False)

        entry = ttk.Entry(bar, textvariable=self.symbol, width=14)
        entry.bind("<Return>", lambda e: self.start_analysis())
        self._field(bar, "Par cripto o ticker", entry, 0)
        self._field(bar, "Tipo", ttk.Combobox(bar, textvariable=self.kind, values=["Auto", "Cripto", "Acción"],
                                               state="readonly", width=8), 1)
        self._field(bar, "Temporalidad", ttk.Combobox(bar, textvariable=self.interval, values=INTERVALS,
                                                       state="readonly", width=6), 2)
        self._field(bar, "Velas de contexto", ttk.Spinbox(bar, from_=64, to=512, textvariable=self.lookback, width=7), 3)
        self._field(bar, "Velas a predecir", ttk.Spinbox(bar, from_=1, to=240, textvariable=self.pred_len, width=7), 4)
        self._field(bar, "Muestras", ttk.Spinbox(bar, from_=1, to=20, textvariable=self.samples, width=5), 5)
        ttk.Checkbutton(bar, text="Backtest", variable=self.backtest).grid(row=1, column=6, padx=(0, 14), pady=(2, 0))
        self.btn = ttk.Button(bar, text="Analizar", style="Accent.TButton", command=self.start_analysis)
        self.btn.grid(row=1, column=7, padx=(0, 10), pady=(2, 0))

        body = ttk.Frame(self.root, padding=(14, 0, 14, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 5.5), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._empty_plot()

        side = ttk.Frame(body, style="Panel.TFrame", padding=12)
        side.grid(row=0, column=1, sticky="ns", padx=(12, 0))
        ttk.Label(side, text="Resultado", style="TLabel", background=PANEL, foreground=FG,
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        self.text = tk.Text(side, width=46, height=20, bg=PANEL, fg=FG, relief="flat", borderwidth=0,
                            font=("Consolas", 10), wrap="word", state="disabled", highlightthickness=0)
        self.text.pack(fill="both", expand=True)
        btns = ttk.Frame(side, style="Panel.TFrame")
        btns.pack(fill="x", pady=(10, 0))
        self.save_csv = ttk.Button(btns, text="Guardar CSV", command=lambda: self.save("csv"), state="disabled")
        self.save_png = ttk.Button(btns, text="Guardar PNG", command=lambda: self.save("png"), state="disabled")
        self.save_csv.pack(side="left", padx=(0, 8))
        self.save_png.pack(side="left")

        foot = ttk.Frame(self.root, padding=(14, 8, 14, 10))
        foot.pack(fill="x")
        self.status = tk.StringVar(value="Listo.")
        ttk.Label(foot, textvariable=self.status).pack(side="left")
        self.progress = ttk.Progressbar(foot, mode="indeterminate", length=140)
        self.progress.pack(side="right")

    # ---- plotting
    def _style_axes(self, ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED)
        ax.grid(True, color="#3a3a3d", linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color("#3c3c3c")

    def _empty_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "Ingresá un par o ticker y presioná Analizar", color=MUTED, ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        self.canvas.draw_idle()

    def _plot(self, r):
        ctx, pred, truth = r["context"], r["pred"], r["truth"]
        n_show = min(len(ctx), 120)
        hist = ctx.tail(n_show).reset_index(drop=True)
        n_h, n_p = len(hist), len(pred)
        xs_h, xs_p = range(n_h), range(n_h, n_h + n_p)
        self.fig.clear()
        ax1, ax2 = self.fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for ax in (ax1, ax2):
            self._style_axes(ax)
        ax1.plot(xs_h, hist["close"], color=HISTORY, linewidth=1.5, label="Histórico")
        ax2.plot(xs_h, hist["volume"], color=HISTORY, linewidth=1.2)
        if truth is not None:
            ax1.plot(xs_p, truth["close"], color=TRUTH, linewidth=1.3, linestyle="--", label="Real")
            ax2.plot(xs_p, truth["volume"], color=TRUTH, linewidth=1.0, linestyle="--")
        ax1.plot(xs_p, pred["close"], color=PREDICTION, linewidth=1.6, label="Predicción")
        ax2.plot(xs_p, pred["volume"], color=PREDICTION, linewidth=1.2)
        ax1.axvline(n_h - 0.5, color="#6b6b6b", linewidth=0.8, linestyle=":")
        ax2.axvline(n_h - 0.5, color="#6b6b6b", linewidth=0.8, linestyle=":")
        ax1.set_ylabel("Cierre", color=MUTED)
        ax2.set_ylabel("Volumen", color=MUTED)
        ax1.set_title(f"{r['symbol']} · {r['interval']}", color=FG, fontsize=12, loc="left")
        leg = ax1.legend(loc="best", facecolor=PANEL, edgecolor="#3c3c3c")
        for t in leg.get_texts():
            t.set_color(FG)
        step = max(n_h // 5, 1)
        ticks = list(range(0, n_h, step)) + [n_h + n_p - 1]
        labels = [hist["timestamps"].iloc[i].strftime("%m-%d %H:%M") for i in range(0, n_h, step)]
        labels.append(pred.index[-1].strftime("%m-%d %H:%M"))
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels, rotation=25, ha="right")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---- actions
    def _set_text(self, s):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", s)
        self.text.configure(state="disabled")

    def start_analysis(self):
        if str(self.btn["state"]) == "disabled":
            return
        symbol = self.symbol.get().strip()
        if not symbol:
            messagebox.showwarning("Kronos", "Ingresá un par cripto (BTCUSDT) o un ticker (MSTR).")
            return
        try:
            args = (symbol, {"Auto": "auto", "Cripto": "crypto", "Acción": "stock"}[self.kind.get()],
                    self.interval.get(), int(self.lookback.get()), int(self.pred_len.get()),
                    int(self.samples.get()), self.backtest.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Kronos", "Velas de contexto, velas a predecir y muestras deben ser números enteros.")
            return
        if not 64 <= args[3] <= 512 or not 1 <= args[4] <= 240 or not 1 <= args[5] <= 20:
            messagebox.showwarning("Kronos", "Rangos válidos: contexto 64-512, predecir 1-240, muestras 1-20.")
            return
        self.btn.configure(state="disabled")
        self.save_csv.configure(state="disabled")
        self.save_png.configure(state="disabled")
        self.progress.start(12)
        threading.Thread(target=self._worker, args=args, daemon=True).start()

    def _worker(self, *args):
        try:
            res = run_analysis(*args, status=lambda m: self.q.put(("status", m)))
            self.q.put(("done", res))
        except Exception as e:
            self.q.put(("error", str(e) or repr(e)))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                    continue
                self.progress.stop()
                self.btn.configure(state="normal")
                if kind == "error":
                    self.status.set("Error.")
                    messagebox.showerror("Kronos", payload)
                else:
                    self.result = payload
                    self._plot(payload)
                    self._set_text(summarize(payload))
                    self.save_csv.configure(state="normal")
                    self.save_png.configure(state="normal")
                    self.status.set("Listo.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def save(self, what):
        r = self.result
        if r is None:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = f"{r['symbol'].lower()}_{r['interval']}_{'backtest' if r['backtest'] else 'forecast'}"
        if what == "csv":
            path = os.path.join(OUTPUT_DIR, base + "_predictions.csv")
            r["pred"].to_csv(path, index_label="timestamps")
        else:
            path = os.path.join(OUTPUT_DIR, base + "_prediction.png")
            self.fig.savefig(path, dpi=150, facecolor=self.fig.get_facecolor())
        self.status.set(f"Guardado: {path}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
