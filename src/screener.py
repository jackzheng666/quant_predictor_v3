"""Strict rule screen. Missing sector evidence disqualifies, absent catalysts do not."""
import numpy as np
import pandas as pd
from config import DEFAULT
from src.indicators import indicators
from src.risk import fee, stop_level, price_limit
from src.data_fetcher import main_board


def screen(histories, metadata, sectors, asof, cfg=DEFAULT):
    candidates, audit = [], []
    target = pd.Timestamp(asof)
    for meta in metadata.to_dict('records'):
        symbol, name = meta['Code'], meta['Name']
        reason = ''
        try:
            if not main_board(symbol) or 'ST' in name.upper() or '退' in name:
                raise ValueError('Excluded board/ST/delisting')
            bars = histories.get(symbol, pd.DataFrame())
            if bars.empty:
                raise ValueError('Missing price history')
            bars = bars[bars.Date <= target]
            if len(bars) < 60:
                raise ValueError('Fewer than 60 observed sessions')
            if bars.iloc[-1].Date != target:
                raise ValueError('Stale/suspended: no bar on screening date')
            if 'Adjustment' not in bars or not bars.Adjustment.eq('raw').all():
                raise ValueError('Screen execution levels require raw prices')
            sector_name = meta.get('Benchmark')
            benchmark = sectors.get(sector_name)
            d = indicators(bars, benchmark)
            r = d.iloc[-1]
            if not np.isfinite(r.sector_relative):
                raise ValueError('Sector benchmark unavailable/unmapped')
            if r.Volume <= 0 or ('TradeStatus' in r and str(r.TradeStatus) == '0'):
                raise ValueError('Suspended')
            if 'IsST' in r and str(r.IsST) in ('1', '1.0'):
                raise ValueError('Historical ST flag')
            if not (r.turnover20 > cfg.min_turnover and r.MA5 > r.MA10 > r.MA20
                    and r.momentum5 > 0 and r.activity > 1 and r.sector_relative > 0):
                raise ValueError('Screen conditions not met')
            if 'PreClose' not in r or not np.isfinite(r.PreClose) or r.PreClose <= 0:
                raise ValueError('Daily price-limit reference unavailable')
            lower, upper = price_limit(r.PreClose)
            if r.Close >= upper or r.Close <= lower:
                raise ValueError('At daily price limit')
            values = [np.clip((r.Close/r.MA20-1)*10,0,1), np.clip((r.activity-1)/2,0,1),
                      np.clip(r.sector_relative*20,0,1), np.nan, np.clip(1-r.volatility,0,1)]
            weights = np.asarray(cfg.weights, dtype=float)
            mask = np.isfinite(values)
            score = float(np.dot(np.asarray(values)[mask], weights[mask])/weights[mask].sum()*100)
            stop = stop_level(r)
            candidates.append(dict(Code=symbol, Name=name, Sector=meta.get('Sector'), Benchmark=sector_name, Date=r.Date,
                Close=r.Close, Score=score, MA5=r.MA5, MA10=r.MA10, MA20=r.MA20, MA60=r.MA60,
                RSI=r.RSI, MACD=r.MACD, ATR14=r.ATR14, Momentum5=r.momentum5,
                Activity=r.activity, SectorRelative=r.sector_relative, EntryTrigger=r.High+.01,
                Support=r.support, Resistance=r.resistance, Stop=stop,
                MinimumCost=100*r.Close+fee(100*r.Close, cfg=cfg),
                Risk='High' if r.volatility > .40 else 'Moderate', Catalyst='Unavailable',
                Adjustment=r.get('Adjustment','unknown'), Source=r.get('Source','unknown'), RetrievedAt=r.get('RetrievedAt','unknown')))
        except (ValueError, KeyError) as exc:
            reason = str(exc)
        audit.append(dict(Code=symbol, Reason=reason or 'Eligible'))
    result = pd.DataFrame(candidates)
    if not result.empty:
        result = result.sort_values(['Score','Code'], ascending=[False,True]).head(10).reset_index(drop=True)
    return result, pd.DataFrame(audit)
