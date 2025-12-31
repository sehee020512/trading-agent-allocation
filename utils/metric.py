import numpy as np

def compute_metrics(equity_df):

    # equity series
    equity = equity_df["equity"].dropna()

    # reteurns
    returns = equity.pct_change().dropna().values
    T = len(returns)

    if T == 0:
        return {"CR": 0, "SR": 0, "MDD": 0, "WR": 0, "Vol": 0}

    # ---- 1. Cumulative Return ----
    CR = equity.iloc[-1] / equity.iloc[0] - 1

    # ---- 2. Mean Return & Volatility ----
    mean_r = np.mean(returns)
    vol = np.std(returns, ddof=1)

    # ---- 3, Sharp Ratio ----
    SR = mean_r / vol if vol > 0 else 0

    # ---- 3. Maximum Drawdown ----
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    MDD = drawdown.max()

    # ---- 4. Win Rate (daily) ----
    WR = np.sum(returns > 0) / T

    return {
        "CR": CR * 100,
        "SR": SR,
        "MDD": MDD * 100,
        "WR": WR * 100,
        "Vol": vol * 100
    }