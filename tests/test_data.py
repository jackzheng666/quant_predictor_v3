import datetime as dt
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from data import get_prices
from test_model import sample


@pytest.mark.parametrize('market,symbol', [('A股', '300308'), ('美股', 'NVDA')])
def test_market_adapters_and_bad_bars(monkeypatch, market, symbol):
    bars = sample(430)
    bars.loc[0, 'High'] = bars.loc[0, 'Low'] - 1
    bars['Volume'] = bars['Volume'].astype(float)
    bars.loc[1, 'Volume'] = np.inf
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs)
        if market == 'A股':
            return bars.rename(columns={'Date':'日期', 'Open':'开盘', 'High':'最高',
                                        'Low':'最低', 'Close':'收盘', 'Volume':'成交量'})
        result = bars.set_index('Date')
        result.columns = pd.MultiIndex.from_product([result.columns, [symbol]])
        return result
    monkeypatch.setitem(sys.modules, 'akshare', SimpleNamespace(stock_zh_a_hist=download))
    monkeypatch.setitem(sys.modules, 'yfinance', SimpleNamespace(download=download))
    result = get_prices(symbol, market, dt.date(2022,1,1), dt.date(2025,1,1))
    assert len(result) == 428
    assert result.Date.is_monotonic_increasing
    assert list(result.columns) == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if market == 'A股':
        assert calls[0]['adjust'] == 'qfq'
    else:
        assert calls[0]['auto_adjust'] is True
