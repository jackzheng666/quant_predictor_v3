"""Provider adapters and SQLite snapshot cache. Errors never become artificial bars."""
import datetime as dt
import hashlib
import json
import re
import sqlite3
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from config import DEFAULT

class DataError(RuntimeError):
    pass


def code(value):
    value = str(value).strip().lower().removeprefix('sh.').removeprefix('sz.')
    if not re.fullmatch(r'\d{6}', value):
        raise ValueError('Stock code must contain six digits')
    return value


def main_board(value):
    return code(value).startswith(('000', '001', '002', '003', '600', '601', '603', '605'))


def normalize(frame):
    d = frame.rename(columns={'日期':'Date','开盘':'Open','最高':'High','最低':'Low','收盘':'Close',
                              '成交量':'Volume','成交额':'Amount', '涨跌幅':'PctChange'}).copy()
    required = ['Date','Open','High','Low','Close','Volume','Amount']
    if not set(required) <= set(d):
        raise DataError('Provider omitted required OHLCV/turnover fields')
    d.Date = pd.to_datetime(d.Date, errors='coerce')
    numeric = [c for c in required if c != 'Date'] + [c for c in ('PreClose','PctChange') if c in d]
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.replace([np.inf, -np.inf], np.nan)
    valid = d[required].notna().all(axis=1) & (d[['Open','High','Low','Close']] > 0).all(axis=1)
    valid &= (d.High >= d[['Open','Close','Low']].max(axis=1)) & (d.Low <= d[['Open','Close']].min(axis=1))
    valid &= (d.Volume >= 0) & (d.Amount >= 0)
    if (~valid).any():
        warnings.warn(f'Discarded {int((~valid).sum())} invalid provider rows', stacklevel=2)
    d = d.loc[valid].drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)
    if d.empty:
        raise DataError('No valid market data returned')
    return d


class MarketData:
    def __init__(self, path='data/market.sqlite', cache_hours=DEFAULT.cache_hours):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = cache_hours
        with sqlite3.connect(self.path) as con:
            con.execute('CREATE TABLE IF NOT EXISTS snapshots (key TEXT PRIMARY KEY, retrieved TEXT, payload TEXT)')

    def cached(self, key, fetch, refresh=False):
        key = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()
        now = pd.Timestamp.now(tz='UTC')
        with sqlite3.connect(self.path) as con:
            hit = con.execute('SELECT retrieved,payload FROM snapshots WHERE key=?', (key,)).fetchone()
        if hit and not refresh and (now-pd.Timestamp(hit[0])).total_seconds() < self.ttl*3600:
            return pd.read_json(__import__('io').StringIO(hit[1]), orient='table')
        result = fetch()
        if result.empty:
            raise DataError('Empty response; cache not updated')
        with sqlite3.connect(self.path) as con:
            con.execute('INSERT OR REPLACE INTO snapshots VALUES (?,?,?)',
                        (key, now.isoformat(), result.to_json(orient='table', date_format='iso')))
        return result

    def bars(self, symbol, start, end, adjust='', refresh=False):
        symbol = code(symbol)
        if adjust not in ('', 'qfq') or pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError('Invalid adjustment or date range')
        # Only completed Shanghai sessions: never cache today\'s partial daily bar.
        today = pd.Timestamp.now(tz='Asia/Shanghai').date()
        end = min(pd.Timestamp(end).date(), today-dt.timedelta(days=1))
        def fetch():
            errors = []
            for provider in ('akshare', 'baostock'):
                try:
                    d = self._ak(symbol, start, end, adjust) if provider == 'akshare' else self._bao(symbol, start, end, adjust)
                    d = normalize(d)
                    d = d[(d.Date >= pd.Timestamp(start)) & (d.Date <= pd.Timestamp(end))].copy()
                    if d.empty:
                        raise DataError('No bars within requested interval')
                    if errors:
                        warnings.warn('Primary provider failed; using backup: '+ '; '.join(errors), stacklevel=2)
                    d['Code'], d['Source'], d['Adjustment'] = symbol, provider, adjust or 'raw'
                    d['RetrievedAt'] = pd.Timestamp.now(tz='UTC').isoformat()
                    return d
                except Exception as exc:
                    errors.append(f'{provider}: {exc}')
            raise DataError('; '.join(errors))
        return self.cached(['bars-v1', symbol, str(start), str(end), adjust], fetch, refresh)

    @staticmethod
    def _ak(symbol, start, end, adjust):
        import akshare as ak
        d = ak.stock_zh_a_hist(symbol=symbol, period='daily', start_date=pd.Timestamp(start).strftime('%Y%m%d'),
                             end_date=pd.Timestamp(end).strftime('%Y%m%d'), adjust=adjust, timeout=15)
        d = normalize(d)
        d.Volume *= 100  # Eastmoney reports lots; canonical volume is shares.
        if not adjust and 'PctChange' in d:
            d['PreClose'] = d.Close/(1+d.PctChange/100)
        return d

    @staticmethod
    def _bao(symbol, start, end, adjust):
        import baostock as bs
        login = bs.login()
        if login.error_code != '0':
            raise DataError(login.error_msg)
        try:
            fields = 'date,open,high,low,close,preclose,volume,amount,tradestatus,isST'
            rs = bs.query_history_k_data_plus(('sh.' if symbol.startswith('6') else 'sz.')+symbol,
                fields, start_date=str(pd.Timestamp(start).date()), end_date=str(pd.Timestamp(end).date()),
                frequency='d', adjustflag='2' if adjust else '3')
            rows = []
            while rs.error_code == '0' and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != '0':
                raise DataError(rs.error_msg)
            d = pd.DataFrame(rows, columns=rs.fields).rename(columns={
                'date':'Date','open':'Open','high':'High','low':'Low','close':'Close',
                'preclose':'PreClose','volume':'Volume','amount':'Amount','tradestatus':'TradeStatus','isST':'IsST'})
            return d
        finally:
            bs.logout()

    @staticmethod
    def _industries():
        import baostock as bs
        login = bs.login()
        if login.error_code != '0':
            raise DataError(login.error_msg)
        try:
            rs = bs.query_stock_industry()
            rows = []
            while rs.error_code == '0' and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != '0':
                raise DataError(rs.error_msg)
            d = pd.DataFrame(rows, columns=rs.fields)
            if d.empty:
                raise DataError('Industry membership response is empty')
            d['Code'] = d.code.map(code)
            return d[['Code','code_name','industry']].rename(columns={'industry':'Sector','code_name':'IndustryName'})
        finally:
            bs.logout()

    def universe(self, count=300, refresh=False):
        """Current liquid main-board snapshot; no manual benchmark mapping needed.
        Backup ranks real daily turnover within HS300+ZZ500 constituents.
        Sector peers are explicitly labeled derived benchmarks, not official indices.
        """
        def fetch():
            import akshare as ak
            try:
                spot = ak.stock_zh_a_spot_em().rename(columns={'代码':'Code','名称':'Name','成交额':'Amount'})
                if spot.empty:
                    raise DataError('Empty AKShare universe')
                source = 'akshare'
            except Exception as exc:
                warnings.warn(f'AKShare universe unavailable; using BaoStock constituents: {exc}', stacklevel=2)
                spot = self._bao_universe()
                source = 'baostock-hs300-zz500'
            spot.Code = spot.Code.map(code)
            spot.Amount = pd.to_numeric(spot.Amount, errors='coerce')
            spot = spot[spot.Code.map(main_board) & ~spot.Name.str.contains('ST|退', case=False, na=True)
                        & (spot.Amount > 0)]
            result = spot.sort_values('Amount',ascending=False).head(count).merge(self._industries(),on='Code',how='left')
            if 'IndustryName' in result:
                result['Name'] = result.IndustryName.fillna(result.Name)
            result['Benchmark'] = result.Sector.map(lambda x: 'peer:'+x if isinstance(x,str) and x.strip() else None)
            result['Source'] = source
            result['RetrievedAt'] = pd.Timestamp.now(tz='UTC').isoformat()
            return result.reset_index(drop=True)
        return self.cached(['universe-auto-v2', count], fetch, refresh)

    @staticmethod
    def _bao_universe():
        import baostock as bs
        login = bs.login()
        if login.error_code != '0':
            raise DataError(login.error_msg)
        try:
            codes = set()
            for query in (bs.query_hs300_stocks, bs.query_zz500_stocks):
                rs = query()
                while rs.error_code == '0' and rs.next():
                    row = dict(zip(rs.fields,rs.get_row_data()))
                    if main_board(row['code']):
                        codes.add(row['code'])
                if rs.error_code != '0':
                    raise DataError(rs.error_msg)
            end = pd.Timestamp.now(tz='Asia/Shanghai').date()-dt.timedelta(days=1)
            begin = end-dt.timedelta(days=20)
            rows = []
            for symbol in sorted(codes):
                rs = bs.query_history_k_data_plus(symbol,'date,close,amount,tradestatus,isST',
                    start_date=str(begin),end_date=str(end),frequency='d',adjustflag='3')
                last = None
                while rs.error_code == '0' and rs.next():
                    last = dict(zip(rs.fields,rs.get_row_data()))
                if rs.error_code != '0':
                    raise DataError(rs.error_msg)
                if last and last['tradestatus']=='1' and last['isST']=='0':
                    rows.append(dict(Code=code(symbol),Name=symbol,Amount=last['amount'],Date=last['date']))
            if not rows:
                raise DataError('No valid BaoStock universe turnover data')
            d = pd.DataFrame(rows)
            latest = d.Date.max()
            return d[d.Date == latest].copy()
        finally:
            bs.logout()

    def trading_dates(self, start, end):
        def fetch():
            import baostock as bs
            login = bs.login()
            if login.error_code != '0':
                raise DataError(login.error_msg)
            try:
                rs = bs.query_trade_dates(start_date=str(pd.Timestamp(start).date()),end_date=str(pd.Timestamp(end).date()))
                rows = []
                while rs.error_code == '0' and rs.next():
                    rows.append(rs.get_row_data())
                if rs.error_code != '0':
                    raise DataError(rs.error_msg)
                d = pd.DataFrame(rows,columns=rs.fields)
                d = d[d.is_trading_day=='1'].copy()
                d['Date'] = pd.to_datetime(d.calendar_date)
                return d[['Date']]
            finally:
                bs.logout()
        return self.cached(['exchange-calendar',str(start),str(end)],fetch).Date

    def sector(self, name, start, end):
        def fetch():
            import akshare as ak
            d = ak.stock_board_industry_hist_em(symbol=name, start_date=pd.Timestamp(start).strftime('%Y%m%d'),
                end_date=pd.Timestamp(end).strftime('%Y%m%d'), period='日k', adjust='')
            d = normalize(d)
            d['Source'], d['RetrievedAt'] = 'akshare-sector', pd.Timestamp.now(tz='UTC').isoformat()
            return d
        return self.cached(['sector-v1', name, str(start), str(end)], fetch)


def peer_benchmarks(histories, metadata, minimum_peers=3):
    """Observed sector-peer equal-weight daily price-return indices, rebased to 100.
    Current membership is biased historically. Missing returns are not invented.
    A row requires >=3 actual observed peer returns; gaps propagate as unavailable.
    """
    result = {}
    if 'Benchmark' not in metadata:
        return result
    calendar = sorted(set(date for b in histories.values() for date in b.Date))
    for name, group in metadata.dropna(subset=['Benchmark']).groupby('Benchmark'):
        if not str(name).startswith('peer:'):
            continue
        series = []
        for symbol in group.Code:
            b = histories.get(symbol)
            if b is not None and not b.empty:
                series.append(b.set_index('Date').Close.rename(symbol))
        if len(series) < minimum_peers:
            continue
        returns = pd.concat(series,axis=1).reindex(calendar).pct_change(fill_method=None)
        daily = returns.mean(axis=1).where(returns.notna().sum(axis=1)>=minimum_peers)
        if not daily.notna().any():
            continue
        # Rebase each observed block independently; future gaps never alter past values.
        blocks = daily.isna().cumsum()
        close = (1+daily).groupby(blocks).cumprod()*100
        result[name] = pd.DataFrame({'Date':daily.index,'Close':close.values,
            'Source':'derived-real-sector-peers','Adjustment':'raw-price-return',
            'RetrievedAt':pd.Timestamp.now(tz='UTC').isoformat()})
    return result
