"""Next-open event simulation. Universe/metadata must be supplied point in time.
Daily bars cannot reconstruct queue fills. Reject limit opens conservatively.
Corporate-action discontinuities freeze fills and invalidate reported performance.
"""
import numpy as np
import pandas as pd
from config import DEFAULT
from src.portfolio import Portfolio
from src.risk import price_limit


def chronological_split(dates, fraction=.7, embargo=5):
    dates = sorted(set(pd.to_datetime(dates)))
    if not 0 < fraction < 1 or embargo < 0:
        raise ValueError('Invalid split')
    cut = int(len(dates)*fraction)
    return dates[:max(0,cut-embargo)], dates[cut:]


def tradable(row, side, cfg=DEFAULT):
    if row is None or row.Volume <= 0 or str(row.get('TradeStatus','1')) in ('0','0.0'):
        return False
    if str(row.get('IsST','0')) in ('1','1.0'):
        return False
    previous = row.get('PreClose', np.nan)
    if not np.isfinite(previous) or previous <= 0:
        return False
    low, high = price_limit(previous, float(row.get('LimitRate',.10)))
    fill = row.Open*(1+cfg.slippage if side=='buy' else 1-cfg.slippage)
    return row.Open < high and fill < high if side=='buy' else row.Open > low and fill > low


def run_backtest(histories, calendar, signal_fn, holding=3, cfg=DEFAULT):
    """signal_fn(date, truncated_histories) -> ranked rows Code/Sector/Stop.
    calendar is an independently supplied exchange/benchmark trading calendar.
    Signals expire after the next calendar session. Pending exits retry each session.
    """
    if holding not in (1,3,5):
        raise ValueError('Holding period must be 1, 3 or 5 trading sessions')
    dates = sorted(set(pd.to_datetime(calendar)))
    if len(dates) < 2:
        raise ValueError('At least two calendar sessions required')
    for bars in histories.values():
        if 'Adjustment' not in bars or not bars.Adjustment.eq('raw').all():
            raise ValueError('Backtest accepts raw execution prices only')
        if bars.Date.duplicated().any() or not bars.Date.is_monotonic_increasing:
            raise ValueError('History dates must be unique and sorted')
    # Same shares/fills for gross and net: remove costs from cashflows only.
    net = Portfolio(cfg.capital)
    gross_cash, gross_positions = cfg.capital, {}
    marks, entered, pending, curve, blocked, actions = {}, {}, [], [], [], []
    frozen = set()
    for i,date in enumerate(dates):
        current = {}
        for code,bars in histories.items():
            match = bars[bars.Date == date]
            if not match.empty:
                r = match.iloc[0]
                current[code] = r
                earlier = bars[bars.Date < date]
                if not earlier.empty and np.isfinite(r.get('PreClose',np.nan)):
                    if abs(r.PreClose/earlier.iloc[-1].Close-1) > .002:
                        frozen.add(code)
                        actions.append(dict(Date=date,Code=code,Reason='Corporate action/reference discontinuity; performance invalid'))
        for code in list(net.positions):
            row = current.get(code)
            p = net.positions[code]
            previous_mark = marks.get(code,p['price'])
            due = i-entered[code] >= holding or previous_mark <= p['stop']
            if due and code not in frozen and tradable(row,'sell',cfg):
                shares = p['shares']
                net.sell(code,row.Open,date,cfg)
                gross_cash += shares*row.Open
                gross_positions.pop(code)
            elif due:
                blocked.append(dict(Date=date,Code=code,Side='SELL',Reason='Suspended/limit/unknown reference/action'))
        for order in pending:
            code = order['Code']
            row = current.get(code)
            if code in frozen or code in net.positions or not tradable(row,'buy',cfg):
                blocked.append(dict(Date=date,Code=code,Side='BUY',Reason='Unfillable next-open order'))
                continue
            if order['Stop'] >= row.Open:
                continue
            if net.buy(code,order['Sector'],row.Open,order['Stop'],date,marks,cfg):
                shares = net.positions[code]['shares']
                gross_cash -= shares*row.Open
                gross_positions[code] = shares
                entered[code] = i
        for code,row in current.items():
            marks[code] = row.Close
        curve.append(dict(Date=date, Net=net.equity(marks),
            Gross=gross_cash+sum(q*marks[c] for c,q in gross_positions.items())))
        truncated = {c:b[b.Date <= date].copy() for c,b in histories.items()}
        signals = signal_fn(date,truncated)
        pending = [] if signals.empty else signals.to_dict('records')
    equity = pd.DataFrame(curve).set_index('Date')
    sells = [x for x in net.ledger if x['Side']=='SELL']
    profits = np.asarray([x['pnl'] for x in sells])
    losses = -profits[profits < 0].sum()
    valid = not actions
    metrics = dict(net_return=equity.Net.iloc[-1]/cfg.capital-1,
        gross_return=equity.Gross.iloc[-1]/cfg.capital-1,
        max_drawdown=float((equity.Net/equity.Net.cummax()-1).min()),
        win_rate=float((profits>0).mean()) if len(profits) else None,
        profit_factor=float(profits[profits>0].sum()/losses) if losses else None,
        closed_trades=len(sells), cost_drag=float(equity.Gross.iloc[-1]-equity.Net.iloc[-1]),
        valid=valid, open_positions=len(net.positions),
        transaction_fees=sum(x['fees'] for x in net.ledger),
        slippage_costs=sum(x['slippage_cost'] for x in net.ledger))
    return dict(equity=equity, metrics=metrics, trades=pd.DataFrame(net.ledger),
                blocked=pd.DataFrame(blocked), actions=pd.DataFrame(actions), account=net)


def benchmark_curve(bars, calendar, capital=10000):
    """Price-return benchmark normalized on first available date; no invented prices."""
    s = bars.set_index('Date').Close.reindex(pd.to_datetime(calendar))
    if s.isna().any():
        raise ValueError('Benchmark missing calendar bars')
    return capital*s/s.iloc[0]
