"""Historical daily bars. Research-only data; provider can revise adjusted history."""
import datetime as dt
import re
import numpy as np
import pandas as pd


def get_prices(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    if start > end:
        raise ValueError('开始日期不能晚于结束日期')
    symbol = symbol.strip().upper()
    if market == 'A股':
        if not re.fullmatch(r'\d{6}', symbol):
            raise ValueError('A股代码请输入6位数字，例如300308')
        import akshare as ak
        bars = ak.stock_zh_a_hist(symbol=symbol, period='daily', start_date=start.strftime('%Y%m%d'),
                                  end_date=end.strftime('%Y%m%d'), adjust='qfq')
        if bars is None or bars.empty:
            raise ValueError('没有下载到A股日线，请检查代码、联网情况或数据源。')
        bars = bars.rename(columns={'日期':'Date','开盘':'Open','最高':'High','最低':'Low','收盘':'Close','成交量':'Volume'})
        bars = bars[['Date','Open','High','Low','Close','Volume']]
    elif market == '美股':
        if not re.fullmatch(r'[A-Z0-9.^=-]{1,18}', symbol):
            raise ValueError('美股代码格式不正确，例如NVDA、TSLA或BRK-B')
        import yfinance as yf
        # Yahoo auto_adjust=True uses split/dividend-adjusted OHLC and is not execution prices.
        bars = yf.download(symbol, start=str(start), end=str(end + dt.timedelta(days=1)),
                           auto_adjust=True, progress=False, threads=False)
        if bars is None or bars.empty:
            raise ValueError('没有下载到美股日线，请检查代码、联网情况或数据源。')
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)
        bars = bars.reset_index()
        bars = bars[['Date','Open','High','Low','Close','Volume']]
    else:
        raise ValueError('仅支持A股或美股')
    bars['Date'] = pd.to_datetime(bars['Date']).dt.tz_localize(None)
    for col in ['Open','High','Low','Close','Volume']:
        bars[col] = pd.to_numeric(bars[col], errors='coerce')
    bars = bars.replace([np.inf, -np.inf], np.nan)
    bars = bars.dropna().drop_duplicates('Date').sort_values('Date').reset_index(drop=True)
    bars = bars[(bars['Open'] > 0) & (bars['Low'] > 0) & (bars['Volume'] >= 0)
                & (bars['High'] >= bars[['Open', 'Close', 'Low']].max(axis=1))
                & (bars['Low'] <= bars[['Open', 'Close']].min(axis=1))]
    if len(bars) < 420:
        raise ValueError(f'有效历史数据只有{len(bars)}行，建议至少420个交易日。')
    return bars.reset_index(drop=True)
