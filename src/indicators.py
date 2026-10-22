"""Backward-looking daily features; sector alignment never fills missing dates."""
import numpy as np
import pandas as pd


def indicators(bars, sector=None):
    d = bars.copy().sort_values('Date').reset_index(drop=True)
    if d.Date.duplicated().any():
        raise ValueError('Duplicate trading dates')
    c = d.Close
    for n in (5, 10, 20, 60):
        d[f'MA{n}'] = c.rolling(n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d['RSI'] = (100 - 100/(1+gain/loss.replace(0, np.nan))).where(loss != 0, 100)
    d.loc[(gain == 0) & (loss == 0), 'RSI'] = 50
    d['MACD'] = c.ewm(span=12, adjust=False).mean()-c.ewm(span=26, adjust=False).mean()
    d['MACD_signal'] = d.MACD.ewm(span=9, adjust=False).mean()
    d['MACD_hist'] = 2*(d.MACD-d.MACD_signal)
    tr = pd.concat([d.High-d.Low, (d.High-c.shift()).abs(), (d.Low-c.shift()).abs()], axis=1).max(axis=1)
    d['ATR14'] = tr.rolling(14).mean()
    d['momentum5'] = c.pct_change(5, fill_method=None)
    d['breakout20'] = c > d.High.shift().rolling(20).max()
    d['turnover20'] = d.Amount.shift().rolling(20).mean()
    d['activity'] = d.Amount/d.turnover20
    d['volatility'] = c.pct_change(fill_method=None).rolling(20).std()*np.sqrt(252)
    d['support'] = d.Low.rolling(20).min()
    d['resistance'] = d.High.shift().rolling(20).max()
    d['sector_relative'] = np.nan
    if sector is not None and not sector.empty:
        prices = sector.sort_values('Date').set_index('Date').Close
        s = prices.pct_change(5, fill_method=None).where(prices.rolling(6).count()==6)
        d['sector_relative'] = d.momentum5 - d.Date.map(s)
    return d
