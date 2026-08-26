"""
vol_surface.py
================
Construye una superficie de volatilidad implícita para una acción, ajustando
el modelo SVI (Stochastic Volatility Inspired, parametrización "raw") a cada
vencimiento disponible.

Flujo:
  1. Descarga la cadena de opciones real desde Yahoo Finance (yfinance).
  2. Calcula precios mid, forward implícito (put-call parity) y tiempo a
     vencimiento para cada contrato.
  3. Invierte Black-Scholes (Brent) para obtener la volatilidad implícita de
     mercado de cada opción OTM (usa puts para K<F y calls para K>=F, que son
     numéricamente más estables).
  4. Ajusta SVI raw: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
     por cada vencimiento, en el espacio de varianza total (w = iv^2 * T)
     vs log-moneyness (k = ln(K/F)).
  5. Chequea arbitraje de calendario básico (varianza total no decreciente en T).
  6. Interpola entre vencimientos (lineal en varianza total) para armar una
     malla continua y grafica: (a) smiles por vencimiento con el ajuste SVI
     superpuesto, y (b) superficie 3D IV(K,T).

Requisitos:
    pip install yfinance numpy scipy pandas matplotlib

Uso:
    python vol_surface.py --ticker AAPL
    python vol_surface.py --ticker TSLA --rate 0.045 --min-volume 5
"""

import argparse
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# 1. Black-Scholes y volatilidad implícita
# ----------------------------------------------------------------------------

def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Precio Black-Scholes-Merton (con dividendo continuo q)."""
    if sigma <= 0 or T <= 0:
        intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, option_type="call", q=0.0):
    """Invierte BS con Brent. Devuelve NaN si el precio no es arbitrable."""
    intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    if price <= intrinsic + 1e-8 or T <= 0:
        return np.nan
    f = lambda sig: bs_price(S, K, T, r, sig, option_type, q) - price
    try:
        # chequeo rápido de signo para evitar excepciones de brentq
        if f(1e-6) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-6, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------
# 2. Descarga y limpieza de la cadena de opciones
# ----------------------------------------------------------------------------

def fetch_option_chain(ticker: str, r: float, q: float, min_volume: int,
                        max_expiries: int) -> pd.DataFrame:
    import time
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"]
    if spot.empty:
        raise RuntimeError(f"No se encontró precio para '{ticker}'. ¿Escribiste bien el ticker?")
    spot = spot.iloc[-1]

    all_expiries = tk.options
    if not all_expiries:
        raise RuntimeError(f"'{ticker}' no tiene opciones listadas en Yahoo Finance.")
    expiries = all_expiries[:max_expiries] if max_expiries else all_expiries

    rows = []
    skipped = []
    for exp in expiries:
        chain = None
        # reintenta ante rate limiting de Yahoo (HTTP 429 / YFRateLimitError)
        for attempt in range(4):
            try:
                chain = tk.option_chain(exp)
                break
            except Exception as e:
                msg = str(e).lower()
                is_rate_limit = ("429" in msg or "rate limit" in msg or
                                  "too many requests" in msg)
                if is_rate_limit and attempt < 3:
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limit de Yahoo en {exp}, reintentando en {wait}s ...")
                    time.sleep(wait)
                    continue
                skipped.append((exp, str(e)))
                break
        if chain is None:
            continue

        T = (pd.Timestamp(exp) - pd.Timestamp.now()).days / 365.0
        if T <= 0:
            continue

        calls, puts = chain.calls.copy(), chain.puts.copy()
        if calls.empty and puts.empty:
            continue
        calls["option_type"], puts["option_type"] = "call", "put"
        both = pd.concat([calls, puts], ignore_index=True)
        both = both[(both["bid"] > 0) & (both["ask"] > 0)]
        if min_volume:
            both = both[both["volume"].fillna(0) >= min_volume]
        if both.empty:
            continue
        both["mid"] = (both["bid"] + both["ask"]) / 2.0
        both["T"] = T
        both["expiry"] = exp
        rows.append(both[["expiry", "T", "strike", "option_type", "mid"]])
        time.sleep(0.3)  # pausa corta entre requests para evitar rate limiting

    if skipped:
        print(f"  Aviso: se saltaron {len(skipped)} vencimiento(s) por error "
              f"({skipped[0][1][:80]}...)" if skipped else "")
    if not rows:
        raise RuntimeError("No se obtuvieron datos de opciones utilizables. Prueba con "
                            "--min-volume 0, otro ticker, o vuelve a correr en unos minutos "
                            "(puede ser rate limiting de Yahoo).")
    df = pd.concat(rows, ignore_index=True)
    df["spot"] = spot
    return estimate_forward_and_iv(df, r, q)


def estimate_forward_and_iv(df: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    """Para cada vencimiento: estima forward vía put-call parity en el strike
    más ATM, luego calcula IV de cada opción OTM."""
    out = []
    for exp, g in df.groupby("expiry"):
        T = g["T"].iloc[0]
        S = g["spot"].iloc[0]
        piv = g.pivot_table(index="strike", columns="option_type", values="mid")
        piv = piv.dropna()
        if piv.empty:
            continue
        # strike más cercano a ATM para estimar el forward: F = K + e^{rT}(C-P)
        # (se usa búsqueda posicional con argmin/iloc en vez de .loc con la
        # etiqueta float, que puede fallar con KeyError por precisión numérica)
        diffs = np.abs(piv.index.values - S)
        pos = int(np.argmin(diffs))
        atm_k = piv.index[pos]
        C, P = piv["call"].iloc[pos], piv["put"].iloc[pos]
        F = atm_k + np.exp(r * T) * (C - P)

        sub = g.copy()
        sub["forward"] = F
        # usa OTM: puts si K<F, calls si K>=F (más líquidas y estables numéricamente)
        sub = sub[((sub["option_type"] == "put") & (sub["strike"] < F)) |
                  ((sub["option_type"] == "call") & (sub["strike"] >= F))]
        sub["iv"] = sub.apply(
            lambda row: implied_vol(row["mid"], S, row["strike"], T, r,
                                     row["option_type"], q), axis=1)
        sub = sub.dropna(subset=["iv"])
        sub["log_moneyness"] = np.log(sub["strike"] / F)
        sub["total_var"] = sub["iv"] ** 2 * T
        out.append(sub)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# 3. Ajuste SVI raw por vencimiento
# ----------------------------------------------------------------------------

@dataclass
class SVISlice:
    expiry: str
    T: float
    params: np.ndarray  # a, b, rho, m, sigma
    k_range: tuple = field(default=(-1.0, 1.0))

    def total_var(self, k):
        a, b, rho, m, sigma = self.params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def iv(self, k):
        return np.sqrt(np.maximum(self.total_var(k), 1e-10) / self.T)


def svi_raw(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi_slice(k, w_market, T) -> np.ndarray:
    """Ajusta (a,b,rho,m,sigma) minimizando error cuadrático en varianza total,
    con las restricciones estándar de no-arbitraje estático de SVI."""
    def resid(x):
        a, b, rho, m, sigma = x
        return svi_raw(k, a, b, rho, m, sigma) - w_market

    x0 = [max(w_market.min(), 1e-4), 0.1, -0.3, 0.0, 0.2]
    # bounds: b>=0 (pendientes no negativas), |rho|<1, sigma>0, a puede ir levemente negativo
    lower = [-1.0, 1e-6, -0.999, -2.0, 1e-4]
    upper = [2.0, 5.0, 0.999, 2.0, 2.0]
    res = least_squares(resid, x0, bounds=(lower, upper), max_nfev=5000)
    return res.x


def fit_all_slices(df: pd.DataFrame) -> list:
    slices = []
    for exp, g in df.groupby("expiry"):
        g = g.sort_values("log_moneyness")
        if len(g) < 5:
            continue  # muy pocos puntos para ajustar 5 parámetros
        T = g["T"].iloc[0]
        params = fit_svi_slice(g["log_moneyness"].values, g["total_var"].values, T)
        slices.append(SVISlice(expiry=exp, T=T, params=params,
                                k_range=(g["log_moneyness"].min(), g["log_moneyness"].max())))
    slices.sort(key=lambda s: s.T)
    return slices


def check_calendar_arbitrage(slices: list, k_grid=None):
    """Chequeo básico: la varianza total SVI en un mismo k debe crecer con T."""
    if k_grid is None:
        k_grid = np.linspace(-0.3, 0.3, 25)
    violations = 0
    for i in range(len(slices) - 1):
        w1 = slices[i].total_var(k_grid)
        w2 = slices[i + 1].total_var(k_grid)
        if np.any(w2 < w1 - 1e-6):
            violations += 1
    if violations:
        print(f"[aviso] posible arbitraje de calendario detectado en {violations} "
              f"par(es) de vencimientos consecutivos (varianza total no monótona).")
    else:
        print("[ok] no se detectó arbitraje de calendario en el chequeo básico.")


# ----------------------------------------------------------------------------
# 3b. SSVI (Surface SVI) — una superficie global en vez de un SVI por slice
# ----------------------------------------------------------------------------
#
# Referencia: Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility
# surfaces". Se usa la parametrización power-law con gamma=1/2:
#
#   phi(theta) = eta / sqrt(theta * (1 + theta))
#   w(k, theta) = (theta/2) * [1 + rho*phi(theta)*k
#                              + sqrt((phi(theta)*k + rho)^2 + (1-rho^2))]
#
# donde theta_t = varianza total ATM del vencimiento t (= IV_atm(t)^2 * t), y
# rho, eta son GLOBALES (compartidos por todos los vencimientos) — eso es lo
# que ata la superficie entera y evita el arbitraje de calendario, en vez de
# solo detectarlo después como en el SVI por slice.
#
# Condición suficiente de no-arbitraje estático (Gatheral-Jacquier, thm 4.2,
# válida para todo theta>0 con gamma=1/2):  eta * (1 + |rho|) <= 2

@dataclass
class SSVISurface:
    rho: float
    eta: float
    gamma: float
    theta_by_T: dict          # {T: theta_t} en los vencimientos observados
    theta_interp: object      # callable T -> theta(T), monótono no-decreciente

    def phi(self, theta):
        return self.eta / (theta ** self.gamma * (1 + theta) ** (1 - self.gamma))

    def total_var(self, k, T):
        theta = self.theta_interp(T)
        theta = np.maximum(theta, 1e-10)
        phi = self.phi(theta)
        return 0.5 * theta * (1 + self.rho * phi * k +
                               np.sqrt((phi * k + self.rho) ** 2 + (1 - self.rho ** 2)))

    def iv(self, k, T):
        return np.sqrt(np.maximum(self.total_var(k, T), 1e-10) / np.maximum(T, 1e-10))

    def is_arbitrage_free(self):
        """Condición suficiente de Gatheral-Jacquier para gamma=1/2."""
        return self.eta * (1 + abs(self.rho)) <= 2.0


def fit_ssvi(df: pd.DataFrame, slices: list, gamma: float = 0.5) -> SSVISurface:
    """Ajusta la superficie SSVI global en dos pasos:
    1) extrae theta_t (varianza total ATM) de cada vencimiento evaluando
       en k=0 el SVI por slice ya ajustado (suaviza ruido de mercado).
    2) ajusta (rho, eta) globalmente contra TODOS los puntos de mercado
       de TODOS los vencimientos a la vez.
    """
    from scipy.interpolate import PchipInterpolator

    theta_by_T = {}
    for sl in slices:
        theta_by_T[sl.T] = float(sl.total_var(np.array([0.0]))[0])

    df = df.copy()
    df["theta_t"] = df["T"].map(theta_by_T)
    df = df.dropna(subset=["theta_t"])

    def resid(x):
        rho, eta = x
        phi = eta / (df["theta_t"].values ** gamma * (1 + df["theta_t"].values) ** (1 - gamma))
        w_model = 0.5 * df["theta_t"].values * (
            1 + rho * phi * df["log_moneyness"].values +
            np.sqrt((phi * df["log_moneyness"].values + rho) ** 2 + (1 - rho ** 2)))
        return w_model - df["total_var"].values

    res = least_squares(resid, x0=[-0.3, 1.0], bounds=([-0.999, 1e-4], [0.999, 10.0]))
    rho_hat, eta_hat = res.x

    Ts = np.array(sorted(theta_by_T.keys()))
    thetas = np.array([theta_by_T[t] for t in Ts])
    # PCHIP preserva monotonía si los datos de entrada ya son monótonos;
    # se fuerza no-decreciente como salvaguarda extra (p.ej. al extrapolar).
    pchip = PchipInterpolator(Ts, thetas, extrapolate=True)

    def theta_interp(T):
        val = pchip(T)
        # aplica acumulado no-decreciente sobre una malla de referencia
        # cuando T es un array, para blindar contra extrapolación no monótona
        if np.ndim(T) > 0:
            order = np.argsort(T)
            val_sorted = np.maximum.accumulate(np.asarray(val)[order])
            out = np.empty_like(val_sorted)
            out[order] = val_sorted
            return out
        return max(val, thetas.min())

    surface = SSVISurface(rho=rho_hat, eta=eta_hat, gamma=gamma,
                           theta_by_T=theta_by_T, theta_interp=theta_interp)

    if surface.is_arbitrage_free():
        print(f"[ok] SSVI global: rho={rho_hat:.4f}, eta={eta_hat:.4f} — cumple la condición "
              f"suficiente de no-arbitraje estático (eta*(1+|rho|)={eta_hat*(1+abs(rho_hat)):.3f} <= 2).")
    else:
        print(f"[aviso] SSVI global: rho={rho_hat:.4f}, eta={eta_hat:.4f} — NO cumple la condición "
              f"suficiente de no-arbitraje estático (eta*(1+|rho|)={eta_hat*(1+abs(rho_hat)):.3f} > 2). "
              f"El ajuste sigue siendo válido como mejor fit, pero revisa la calidad de los datos "
              f"de entrada (spreads muy anchos, pocos puntos, etc.).")
    return surface


# ----------------------------------------------------------------------------
# 4. Gráficos
# ----------------------------------------------------------------------------

def plot_smiles(df: pd.DataFrame, slices: list, ticker: str, outpath: str):
    import matplotlib.pyplot as plt
    n = len(slices)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, sl in enumerate(slices):
        ax = axes[i // ncols][i % ncols]
        g = df[df["expiry"] == sl.expiry]
        ax.scatter(g["log_moneyness"], g["iv"] * 100, s=15, color="tab:blue",
                   label="mercado", zorder=3)
        k_line = np.linspace(*sl.k_range, 200)
        ax.plot(k_line, sl.iv(k_line) * 100, color="tab:red", lw=2, label="SVI")
        ax.set_title(f"{sl.expiry}  (T={sl.T:.2f}a)")
        ax.set_xlabel("log-moneyness k=ln(K/F)")
        ax.set_ylabel("IV (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{ticker}: smiles de IV por vencimiento (mercado vs SVI)", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_surface(slices: list, ticker: str, outpath: str, n_k=60, n_t=60):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    k_min = max(s.k_range[0] for s in slices)
    k_max = min(s.k_range[1] for s in slices)
    k_grid = np.linspace(k_min, k_max, n_k)
    T_fit = np.array([s.T for s in slices])
    t_grid = np.linspace(T_fit.min(), T_fit.max(), n_t)

    # varianza total interpolada linealmente en T para cada k, usando los ajustes SVI
    W = np.array([s.total_var(k_grid) for s in slices])  # (n_slices, n_k)
    W_interp = np.empty((n_t, n_k))
    for j in range(n_k):
        W_interp[:, j] = np.interp(t_grid, T_fit, W[:, j])
    IV = np.sqrt(np.maximum(W_interp, 1e-10) / t_grid[:, None])

    K_mesh, T_mesh = np.meshgrid(k_grid, t_grid)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_mesh, T_mesh, IV * 100, cmap="viridis",
                            linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("log-moneyness k=ln(K/F)")
    ax.set_ylabel("T (años)")
    ax.set_zlabel("IV (%)")
    ax.set_title(f"{ticker}: superficie de volatilidad implícita (SVI por slice)")
    fig.colorbar(surf, shrink=0.6, aspect=12, label="IV (%)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_ssvi_smiles(df: pd.DataFrame, slices: list, surface: "SSVISurface",
                      ticker: str, outpath: str):
    """Compara, por vencimiento: mercado vs SVI-por-slice vs SSVI global."""
    import matplotlib.pyplot as plt
    n = len(slices)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, sl in enumerate(slices):
        ax = axes[i // ncols][i % ncols]
        g = df[df["expiry"] == sl.expiry]
        ax.scatter(g["log_moneyness"], g["iv"] * 100, s=15, color="tab:blue",
                   label="mercado", zorder=3)
        k_line = np.linspace(*sl.k_range, 200)
        ax.plot(k_line, sl.iv(k_line) * 100, color="tab:red", lw=1.5, ls="--",
                label="SVI (por slice)")
        ax.plot(k_line, surface.iv(k_line, sl.T) * 100, color="tab:green", lw=2,
                label="SSVI (global)")
        ax.set_title(f"{sl.expiry}  (T={sl.T:.2f}a)")
        ax.set_xlabel("log-moneyness k=ln(K/F)")
        ax.set_ylabel("IV (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{ticker}: SVI por slice vs SSVI global", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_ssvi_surface(surface: "SSVISurface", slices: list, ticker: str, outpath: str,
                       n_k=60, n_t=60):
    """Superficie 3D construida 100% con SSVI (globalmente consistente,
    sin interpolar entre slices independientes como en plot_surface)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    k_min = max(s.k_range[0] for s in slices)
    k_max = min(s.k_range[1] for s in slices)
    k_grid = np.linspace(k_min, k_max, n_k)
    T_fit = np.array(sorted(surface.theta_by_T.keys()))
    t_grid = np.linspace(T_fit.min(), T_fit.max(), n_t)

    IV = np.array([surface.iv(k_grid, T) for T in t_grid])  # (n_t, n_k)

    K_mesh, T_mesh = np.meshgrid(k_grid, t_grid)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_mesh, T_mesh, IV * 100, cmap="viridis",
                            linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("log-moneyness k=ln(K/F)")
    ax.set_ylabel("T (años)")
    ax.set_zlabel("IV (%)")
    afree = "sin arbitraje estático (cond. suficiente)" if surface.is_arbitrage_free() else \
            "condición de no-arbitraje NO verificada"
    ax.set_title(f"{ticker}: superficie SSVI global — ρ={surface.rho:.3f}, η={surface.eta:.3f} ({afree})")
    fig.colorbar(surf, shrink=0.6, aspect=12, label="IV (%)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Superficie de volatilidad implícita con SVI")
    ap.add_argument("--ticker", required=True, help="Ticker, ej. AAPL, TSLA")
    ap.add_argument("--rate", type=float, default=0.045, help="Tasa libre de riesgo anual (flat)")
    ap.add_argument("--dividend", type=float, default=0.0, help="Dividend yield continuo")
    ap.add_argument("--min-volume", type=int, default=1, help="Volumen mínimo por contrato")
    ap.add_argument("--max-expiries", type=int, default=8, help="Número máximo de vencimientos a usar")
    ap.add_argument("--outdir", default=".", help="Carpeta de salida para los gráficos")
    args = ap.parse_args()

    print(f"Descargando cadena de opciones de {args.ticker} ...")
    df = fetch_option_chain(args.ticker, args.rate, args.dividend,
                             args.min_volume, args.max_expiries)
    print(f"{len(df)} cotizaciones OTM válidas en {df['expiry'].nunique()} vencimientos.")

    print("Ajustando SVI por vencimiento ...")
    slices = fit_all_slices(df)
    if len(slices) < 2:
        raise RuntimeError("Se necesitan al menos 2 vencimientos con ajuste válido para la superficie.")

    check_calendar_arbitrage(slices)

    smiles_path = f"{args.outdir}/{args.ticker}_smiles.png"
    surface_path = f"{args.outdir}/{args.ticker}_surface.png"
    print("Graficando smiles (SVI por slice) ...")
    plot_smiles(df, slices, args.ticker, smiles_path)
    print("Graficando superficie 3D (SVI por slice) ...")
    plot_surface(slices, args.ticker, surface_path)

    print("\nParámetros SVI por vencimiento:")
    params_df = pd.DataFrame(
        [[s.expiry, round(s.T, 3), *np.round(s.params, 4)] for s in slices],
        columns=["expiry", "T", "a", "b", "rho", "m", "sigma"])
    print(params_df.to_string(index=False))
    params_df.to_csv(f"{args.outdir}/{args.ticker}_svi_params.csv", index=False)

    print("\nAjustando SSVI global (superficie única, todos los vencimientos atados) ...")
    surface = fit_ssvi(df, slices)

    ssvi_smiles_path = f"{args.outdir}/{args.ticker}_ssvi_smiles.png"
    ssvi_surface_path = f"{args.outdir}/{args.ticker}_ssvi_surface.png"
    print("Graficando smiles SVI-por-slice vs SSVI global ...")
    plot_ssvi_smiles(df, slices, surface, args.ticker, ssvi_smiles_path)
    print("Graficando superficie SSVI 3D ...")
    plot_ssvi_surface(surface, slices, args.ticker, ssvi_surface_path)

    ssvi_params_df = pd.DataFrame([{
        "rho": round(surface.rho, 4), "eta": round(surface.eta, 4),
        "gamma": surface.gamma,
        "arbitrage_free_sufficient_cond": surface.is_arbitrage_free(),
    }])
    ssvi_params_df.to_csv(f"{args.outdir}/{args.ticker}_ssvi_params.csv", index=False)

    print(f"\nListo. Archivos generados:\n  {smiles_path}\n  {surface_path}\n"
          f"  {args.outdir}/{args.ticker}_svi_params.csv\n"
          f"  {ssvi_smiles_path}\n  {ssvi_surface_path}\n"
          f"  {args.outdir}/{args.ticker}_ssvi_params.csv")


if __name__ == "__main__":
    main()
