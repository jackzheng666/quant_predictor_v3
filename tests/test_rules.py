"""Synthetic fixtures verify mechanics only, never investment performance."""
import pandas as pd
import numpy as np
import pytest
from config import DEFAULT
from src.data_fetcher import MarketData, DataError, normalize
from src.indicators import indicators
from src.screener import screen
from src.risk import size_position, fee, price_limit
from src.portfolio import Portfolio
from src.backtest import run_backtest, chronological_split, benchmark_curve


def bars(n=90):
    c = 10+np.arange(n)*.03
    d = pd.DataFrame(dict(Date=pd.bdate_range('2024-01-02',periods=n),Open=c,High=c+.1,Low=c-.1,
        Close=c,Volume=30_000_000.,Amount=400_000_000.,Adjustment='raw',Source='fixture'))
    d['PreClose'] = d.Close.shift().fillna(c[0])
    d.loc[n-1,'Amount'] = 800_000_000
    return d


def test_indicator_causality_and_previous_windows():
    b = bars()
    sector = b[['Date','Close']].assign(Close=10.)
    x = indicators(b,sector)
    changed = b.copy()
    changed.loc[70:,'Close'] *= 2
    y = indicators(changed,sector)
    pd.testing.assert_frame_equal(x.iloc[:70],y.iloc[:70])
    assert x.iloc[-1].turnover20 == 400_000_000
    assert x.iloc[-1].activity == 2
    assert x.iloc[-1].RSI == 100
    assert x.iloc[-1].sector_relative > 0
    assert x.iloc[-1].MA60 == pytest.approx(b.Close.tail(60).mean())


def test_screener_missing_sector_and_normalized_score():
    b = bars()
    meta = pd.DataFrame([dict(Code='600000',Name='测试',Sector='银行',Benchmark='银行')])
    empty,audit = screen({'600000':b},meta,{},b.Date.iloc[-1])
    assert empty.empty and 'unavailable' in audit.Reason.iloc[0]
    sector = b[['Date','Close']].assign(Close=10.)
    ranked,_ = screen({'600000':b},meta,{'银行':sector},b.Date.iloc[-1])
    assert len(ranked) == 1
    assert ranked.Catalyst.iloc[0] == 'Unavailable'
    assert 0 <= ranked.Score.iloc[0] <= 100
    # Missing catalyst weight is removed, not filled with a guessed score.
    r = indicators(b,sector).iloc[-1]
    expected = (np.clip((r.Close/r.MA20-1)*10,0,1)*25 + .5*25 +
                np.clip(r.sector_relative*20,0,1)*20 + np.clip(1-r.volatility,0,1)*15)/85*100
    assert ranked.Score.iloc[0] == pytest.approx(expected)
    stale,_ = screen({'600000':b.iloc[:-1]},meta,{'银行':sector},b.Date.iloc[-1])
    assert stale.empty
    meta.loc[0,'Name']='*ST测试'
    assert screen({'600000':b},meta,{'银行':sector},b.Date.iloc[-1])[0].empty


def test_cache_separates_raw_adjusted_and_fallback(monkeypatch,tmp_path):
    store = MarketData(tmp_path/'bars.db')
    calls=[]
    def download(symbol,start,end,adjust):
        calls.append(adjust)
        return bars()
    monkeypatch.setattr(store,'_ak',download)
    for _ in range(2):
        result = store.bars('sh.600000','2024-01-01','2024-06-01')
        assert result.Adjustment.eq('raw').all()
        assert result.RetrievedAt.notna().all()
    store.bars('600000','2024-01-01','2024-06-01','qfq')
    assert calls == ['', 'qfq']
    monkeypatch.setattr(store,'_ak',lambda *a: (_ for _ in ()).throw(DataError('primary offline')))
    monkeypatch.setattr(store,'_bao',download)
    with pytest.warns(UserWarning,match='Primary provider failed'):
        assert store.bars('600000','2024-01-01','2024-06-01',refresh=True).Source.eq('baostock').all()
    monkeypatch.setattr(store,'_bao',lambda *a: (_ for _ in ()).throw(DataError('backup offline')))
    with pytest.raises(DataError,match='primary offline.*backup offline'):
        store.bars('600000','2024-01-01','2024-06-01',refresh=True)


def test_normalize_invalid_and_duplicates():
    b = bars(5)
    b.loc[0,'High']=0
    with pytest.warns(UserWarning):
        result = normalize(pd.concat([b,b.iloc[[1]]]))
    assert len(result)==4
    assert result.Date.is_unique


def test_risk_lots_fees_concentration():
    order = size_position(10,9.8,10000)
    assert order['shares']%100==0
    assert order['cost']<=3000
    assert order['estimated_loss']<=100
    assert size_position(40,39,10000)['shares']==0
    assert size_position(10,9.8,10000,sector_value=5000)['shares']==0
    assert fee(1000,True)>fee(1000)
    assert price_limit(10.05)==(9.05,11.06)
    with pytest.raises(ValueError):
        size_position(10,11,10000)


def test_paper_tplusone():
    p = Portfolio()
    assert p.buy('600000','银行',10,9.8,'2024-01-02',{})
    with pytest.raises(ValueError,match=r'T\+1'):
        p.sell('600000',10,'2024-01-02')
    p.sell('600000',10,'2024-01-03')
    assert p.cash<10000 and not p.positions


def test_backtest_next_open_and_calendar_exit_delay():
    b = bars(8)
    b.loc[1,'Open']=10.10
    b.loc[1,'High']=10.20
    # Due exit after one exchange session; suspension then limit-down defer it.
    b.loc[2,'Volume']=0
    b.loc[3,'Open']=price_limit(b.loc[3,'PreClose'])[0]
    b.loc[3,'Low']=b.loc[3,'Open']
    origin = b.Date.iloc[0]
    def signal(date,hist):
        assert all((x.Date<=date).all() for x in hist.values())
        return pd.DataFrame([dict(Code='600000',Sector='银行',Stop=9.8)]) if date==origin else pd.DataFrame()
    result = run_backtest({'600000':b},b.Date,signal,holding=1)
    t = result['trades']
    assert list(t.Side)==['BUY','SELL']
    assert t.Date.iloc[0]==b.Date.iloc[1]
    assert t.Date.iloc[1]==b.Date.iloc[4]
    assert len(result['blocked'])==2
    assert result['metrics']['gross_return']>result['metrics']['net_return']
    assert result['metrics']['valid']


def test_limit_buy_and_missing_reference_rejected():
    b=bars(5)
    def signal(date,hist):
        return pd.DataFrame([dict(Code='600000',Sector='银行',Stop=9.8)])
    b.loc[1:,'PreClose']=np.nan
    result=run_backtest({'600000':b},b.Date,signal)
    assert result['trades'].empty
    assert result['metrics']['net_return']==0
    b=bars(5)
    b.loc[1,'Open']=price_limit(b.loc[1,'PreClose'])[1]
    result=run_backtest({'600000':b},b.Date,lambda d,h:signal(d,h) if d==b.Date.iloc[0] else pd.DataFrame())
    assert result['trades'].empty


def test_actions_invalidate_and_adjusted_rejected():
    b=bars(5)
    b.loc[2,'PreClose']=5.
    result=run_backtest({'600000':b},b.Date,lambda d,h:pd.DataFrame())
    assert not result['metrics']['valid']
    b.Adjustment='qfq'
    with pytest.raises(ValueError,match='raw'):
        run_backtest({'600000':b},b.Date,lambda d,h:pd.DataFrame())


def test_split_embargo_and_benchmark():
    b=bars(100)
    train,test=chronological_split(b.Date)
    assert train[-1]==b.Date.iloc[64] and test[0]==b.Date.iloc[70]
    assert benchmark_curve(b,b.Date).iloc[0]==10000
    with pytest.raises(ValueError,match='missing'):
        benchmark_curve(b.iloc[1:],b.Date)


def test_dashboard_empty_and_paper_pages():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(__import__('pathlib').Path(__file__).resolve().parents[1]/'app.py')).run()
    assert not app.exception
    for page in ['个股分析','组合模拟','策略表现']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception


def test_dashboard_real_data_flow_with_mocked_provider(monkeypatch,tmp_path):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    b=bars()
    b['RetrievedAt']='2026-10-14T00:00:00+00:00'
    meta=pd.DataFrame([dict(Code='600000',Name='测试',Sector='银行',Benchmark='银行')])
    monkeypatch.setattr(MarketData,'universe',lambda *a,**k:meta)
    monkeypatch.setattr(MarketData,'bars',lambda self,*a,**k:b.copy())
    monkeypatch.setattr(MarketData,'sector',lambda self,*a,**k:b[['Date','Close']].assign(Close=10.))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
    next(x for x in app.button if x.label=='获取300只主板股票池').click().run()
    next(x for x in app.button if x.label=='自动扫描300只主板股票 / Top 10').click().run()
    assert not app.exception and not app.error
    app.sidebar.radio[0].set_value('每日 Top 10').run()
    assert not app.exception
    assert len(app.dataframe[0].value)==1
    app.sidebar.radio[0].set_value('个股分析').run()
    next(x for x in app.button if x.label=='Fetch Market Data').click().run()
    assert not app.exception and len(app.get('plotly_chart'))==5


def test_backtest_future_changes_do_not_affect_earlier_fills():
    b=bars(10)
    def signal(date,hist):
        return pd.DataFrame([dict(Code='600000',Sector='银行',Stop=9.8)]) if len(hist['600000'])==1 else pd.DataFrame()
    original=run_backtest({'600000':b},b.Date,signal,1)
    changed=b.copy()
    changed.loc[5:,['Open','High','Low','Close','PreClose']]*=1.5
    later=run_backtest({'600000':changed},changed.Date,signal,1)
    pd.testing.assert_frame_equal(original['trades'],later['trades'])
    pd.testing.assert_frame_equal(original['equity'].iloc[:5],later['equity'].iloc[:5])


def test_peer_benchmarks_observed_returns_and_future_gaps():
    from src.data_fetcher import peer_benchmarks
    meta=pd.DataFrame([dict(Code=c,Benchmark='peer:银行') for c in ['600000','600001','600002']])
    histories={c:bars(90) for c in meta.Code}
    peer=peer_benchmarks(histories,meta)['peer:银行']
    r=indicators(histories['600000'],peer)
    assert r.sector_relative.iloc[-1]==pytest.approx(0,abs=1e-12)
    assert peer.Source.eq('derived-real-sector-peers').all()
    modified={c:b.drop(index=70).copy() for c,b in histories.items()}
    later=peer_benchmarks(modified,meta)['peer:银行']
    pd.testing.assert_series_equal(peer.Close.iloc[:70],later.Close.iloc[:70])
    assert peer_benchmarks({'600000':bars()},meta)=={}


def test_automatic_universe_primary_and_backup(monkeypatch,tmp_path):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'akshare',SimpleNamespace(stock_zh_a_spot_em=lambda:pd.DataFrame({
        '代码':['600000','000001','300001','600001'],'名称':['浦发银行','平安银行','创业板','ST公司'],
        '成交额':[300e6,400e6,500e6,600e6]})))
    store=MarketData(tmp_path/'cache.db')
    monkeypatch.setattr(store,'_industries',lambda:pd.DataFrame({'Code':['600000','000001'],'Sector':['银行','银行']}))
    result=store.universe()
    assert list(result.Code)==['000001','600000']
    assert result.Benchmark.eq('peer:银行').all()
    monkeypatch.setitem(sys.modules,'akshare',SimpleNamespace(stock_zh_a_spot_em=lambda:(_ for _ in ()).throw(DataError('offline'))))
    monkeypatch.setattr(store,'_bao_universe',lambda:pd.DataFrame({'Code':['600000'],'Name':['浦发银行'],'Amount':[300e6]}))
    with pytest.warns(UserWarning,match='BaoStock'):
        result=store.universe(refresh=True)
    assert result.Source.eq('baostock-hs300-zz500').all()


def test_single_fetch_keeps_raw_when_adjusted_fails(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    b=bars()
    b['RetrievedAt']='2026-10-14T00:00:00+00:00'
    def download(self,symbol,start,end,adjust=''):
        if adjust:
            raise DataError('adjusted unavailable')
        return b
    monkeypatch.setattr(MarketData,'bars',download)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
    next(x for x in app.button if x.label=='Fetch Market Data').click().run()
    assert not app.exception
    assert app.warning and len(app.get('plotly_chart'))==5
    assert any('最新交易日' in x.value for x in app.caption)


def test_screen_button_auto_fetches_without_csv(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    meta=pd.DataFrame([dict(Code=c,Name=c,Sector='银行',Benchmark='peer:银行')
                       for c in ['600000','600001','600002']])
    monkeypatch.setattr(MarketData,'universe',lambda *a,**k:meta)
    monkeypatch.setattr(MarketData,'bars',lambda *a,**k:bars())
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
    next(x for x in app.button if x.label=='自动扫描300只主板股票 / Top 10').click().run()
    assert not app.exception and not app.error
    app.sidebar.radio[0].set_value('每日 Top 10').run()
    assert not app.exception
    assert len(app.dataframe[1].value)==3
