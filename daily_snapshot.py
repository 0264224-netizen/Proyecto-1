"""
daily_snapshot.py
==================
Version ligera de vol_surface.py pensada para correr sola, todos los dias,
en GitHub Actions (o cualquier cron). No genera graficos (para ahorrar tiempo
de ejecucion); solo descarga la cadena de opciones, ajusta SVI + SSVI, y
agrega una fila al historial CSV por ticker.

Uso:
    python daily_snapshot.py --tickers AAPL,TSLA,MSFT --history-dir history

Se disena para correr en GitHub Actions: ver .github/workflows/daily_snapshot.yml
"""

import argparse
import datetime
import os
import sys
import traceback

import pandas as pd

# reutiliza toda la logica ya probada de vol_surface.py
import vol_surface as vs


def append_snapshot(snapshot_file: str, ticker: str, date: str, surface) -> pd.DataFrame:
    rows = [{"ticker": ticker, "date": date, "T": T, "theta": theta,
             "rho": surface.rho, "eta": surface.eta}
            for T, theta in surface.theta_by_T.items()]
    new_df = pd.DataFrame(rows)

    if os.path.exists(snapshot_file):
        hist = pd.read_csv(snapshot_file)
        hist = hist[~((hist["ticker"] == ticker) & (hist["date"] == date))]
        hist = pd.concat([hist, new_df], ignore_index=True)
    else:
        hist = new_df
    hist.to_csv(snapshot_file, index=False)
    return hist


def run_ticker(ticker: str, history_dir: str, rate: float, dividend: float,
               min_volume: int, max_expiries: int, today: str) -> bool:
    """Devuelve True si el snapshot de hoy se guardo bien, False si fallo."""
    try:
        df = vs.fetch_option_chain(ticker, rate, dividend, min_volume, max_expiries)
        slices = vs.fit_all_slices(df)
        if len(slices) < 2:
            print(f"[{ticker}] pocos vencimientos validos, se salta.")
            return False
        surface = vs.fit_ssvi(df, slices)
        snapshot_file = os.path.join(history_dir, f"{ticker}_ssvi_history.csv")
        hist = append_snapshot(snapshot_file, ticker, today, surface)
        print(f"[{ticker}] snapshot guardado para {today}. "
              f"Dias acumulados: {hist[hist['ticker']==ticker]['date'].nunique()}")
        return True
    except Exception as e:
        print(f"[{ticker}] ERROR: {e}")
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(description="Snapshot diario de superficies SSVI")
    ap.add_argument("--tickers", required=True, help="Lista separada por comas, ej. AAPL,TSLA")
    ap.add_argument("--history-dir", default="history", help="Carpeta donde se guardan los CSV")
    ap.add_argument("--rate", type=float, default=0.045)
    ap.add_argument("--dividend", type=float, default=0.0)
    ap.add_argument("--min-volume", type=int, default=1)
    ap.add_argument("--max-expiries", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.history_dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    print(f"Corriendo snapshot diario para {len(tickers)} ticker(s): {tickers}  ({today})")
    results = {}
    for ticker in tickers:
        results[ticker] = run_ticker(ticker, args.history_dir, args.rate, args.dividend,
                                      args.min_volume, args.max_expiries, today)

    ok = sum(results.values())
    print(f"\nResumen: {ok}/{len(tickers)} tickers guardados correctamente.")
    # no falla el job completo si un ticker individual falla (ej. rate limit puntual);
    # solo falla si TODOS fallaron, para que GitHub marque el run como error real.
    if ok == 0 and tickers:
        sys.exit(1)


if __name__ == "__main__":
    main()
