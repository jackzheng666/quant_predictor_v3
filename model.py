"""Causal features and non-overlapping rolling-origin evaluation for a 5-day horizon."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HORIZON = 5
FEATURES = ['ret1','ret5','ret10','ret20','range1','gap1','vol10','vol20','atr14',
            'ma5_gap','ma10_gap','ma20_gap','ma60_gap','rsi14','volume_ratio','vol_change','close_position']
QUANTILES = (0.1, 0.5, 0.9)
TARGETS = {'close': 'target', 'highest': 'target_highest', 'lowest': 'target_lowest'}


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy().reset_index(drop=True)
    if df.empty or not pd.to_datetime(df['Date']).is_monotonic_increasing or df['Date'].duplicated().any():
        raise ValueError('行情日期必须唯一并按时间升序排列')
    c,h,l,o,v = [df[k].astype(float) for k in ('Close','High','Low','Open','Volume')]
    for n in [1,5,10,20]:
        df[f'ret{n}'] = c.pct_change(n)
    df['range1'] = (h-l)/c
    df['gap1'] = o/c.shift(1)-1
    df['vol10'] = c.pct_change().rolling(10).std()
    df['vol20'] = c.pct_change().rolling(20).std()
    true_range = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr14'] = true_range.rolling(14).mean()/c
    for n in [5,10,20,60]:
        df[f'ma{n}_gap'] = c/c.rolling(n).mean()-1
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    df['rsi14'] = 100-100/(1+gain/(loss+1e-12))
    df['volume_ratio'] = v/(v.rolling(20).mean()+1e-12)
    df['vol_change'] = v.pct_change().replace([np.inf,-np.inf],np.nan)
    df['close_position'] = (c-l)/(h-l).replace(0,np.nan)
    # Predict close-to-close 5-session return. NEVER train on last five unlabeled rows.
    df['target'] = c.shift(-HORIZON)/c-1
    # All five FUTURE sessions must exist; the origin bar is excluded.
    df['target_highest'] = pd.concat([h.shift(-i) for i in range(1, HORIZON+1)], axis=1).max(axis=1, skipna=False)/c-1
    df['target_lowest'] = pd.concat([l.shift(-i) for i in range(1, HORIZON+1)], axis=1).min(axis=1, skipna=False)/c-1
    df[FEATURES] = df[FEATURES].replace([np.inf,-np.inf],np.nan)
    return df


def model_for(q: float) -> LGBMRegressor:
    return LGBMRegressor(objective='quantile', alpha=q, n_estimators=180, learning_rate=0.035,
                        num_leaves=15, max_depth=4, min_child_samples=25,
                        subsample=1.0, colsample_bytree=0.9, reg_lambda=2.0,
                        random_state=42, verbosity=-1, n_jobs=2)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str = 'target') -> np.ndarray:
    if train.empty or test.empty or train[target].isna().any():
        raise ValueError('训练数据不完整或为空')
    arr = np.column_stack([model_for(q).fit(train[FEATURES],train[target]).predict(test[FEATURES])
                           for q in QUANTILES])
    # Quantile crossing correction; preserves monotonic displayed interval.
    return np.sort(arr, axis=1)


@dataclass
class Forecast:
    low: float
    median: float
    high: float
    close: float
    date: pd.Timestamp
    count: int
    results: pd.DataFrame
    features: pd.DataFrame
    quantiles: dict

    def prices(self, target: str) -> np.ndarray:
        return self.close * (1 + np.asarray(self.quantiles[target]))


def run_research(bars: pd.DataFrame, backtest_points: int = 24, train_window: int = 504) -> Forecast:
    if backtest_points < 0 or train_window < 200:
        raise ValueError('回测期数不能为负，训练窗口至少200日')
    df = make_features(bars)
    last = len(df)-1
    if df.iloc[[last]][FEATURES].isna().any().any():
        raise ValueError('最新交易日特征不完整，无法预测')

    def training(origin):
        # Row t has an outcome at t+5. Only outcomes strictly before origin
        # are admitted; filtering features never changes trading-session offsets.
        return df.iloc[max(0, origin-HORIZON-train_window):origin-HORIZON].dropna(
            subset=FEATURES + list(TARGETS.values()))

    def predictions(train, test):
        values = {name: fit_predict(train, test, target)[0]
                  for name, target in TARGETS.items()}
        # Enforce the known ordering lowest <= fifth close <= highest at each
        # quantile. Marginal quantiles are still evaluated separately below.
        values['lowest'] = np.minimum(values['lowest'], values['close'])
        values['highest'] = np.maximum(values['highest'], values['close'])
        return values

    records = []
    origins = range(max(0, last-HORIZON*backtest_points), last-HORIZON+1, HORIZON)
    for idx in origins:
        test = df.iloc[[idx]]
        train = training(idx)
        if len(train) < 200 or test[FEATURES + list(TARGETS.values())].isna().any().any():
            continue
        pred = predictions(train, test)
        record = {'Date': test.iloc[0]['Date'], 'price': float(test.iloc[0]['Close']),
                  'outcome_date': df.iloc[idx+HORIZON]['Date'],
                  'train_end_date': train.iloc[-1]['Date'],
                  'train_label_end_date': df.iloc[int(train.index[-1])+HORIZON]['Date']}
        for name, target in TARGETS.items():
            prefix = '' if name == 'close' else name + '_'
            record.update({prefix + f'q{int(q*100)}': float(value)
                           for q, value in zip(QUANTILES, pred[name])})
            record[prefix + 'actual'] = float(test.iloc[0][target])
        records.append(record)
    now_train = training(last)
    if len(now_train) < 200:
        raise ValueError('训练数据过少')
    quantiles = predictions(now_train, df.iloc[[last]])
    low, med, high = quantiles['close']
    return Forecast(float(low), float(med), float(high), float(df.iloc[last]['Close']),
                    df.iloc[last]['Date'], len(now_train), pd.DataFrame(records), df, quantiles)


def levels(bars: pd.DataFrame, lookback: int=60) -> dict:
    """Heuristic backward-looking price zones, NOT quantile prediction intervals."""
    sample=bars.tail(lookback)
    last=float(bars.iloc[-1]['Close'])
    h=sample['High'].astype(float).to_numpy()
    l=sample['Low'].astype(float).to_numpy()
    # Rolling extrema + nearby past turning points; return intervals with ATR-derived width.
    tr=pd.concat([bars['High']-bars['Low'], (bars['High']-bars['Close'].shift()).abs(),
                  (bars['Low']-bars['Close'].shift()).abs()],axis=1).max(axis=1)
    atr=float(tr.tail(14).mean())
    width=max(0.35*atr, last*0.003)
    lows=[l[i] for i in range(2,len(l)-2) if l[i]==min(l[i-2:i+3]) and l[i]<last]
    highs=[h[i] for i in range(2,len(h)-2) if h[i]==max(h[i-2:i+3]) and h[i]>last]
    below=np.array(lows if lows else [float(np.min(l))]); above=np.array(highs if highs else [float(np.max(h))])
    support=float(np.max(below[below<last])) if np.any(below<last) else float(np.min(l))
    resistance=float(np.min(above[above>last])) if np.any(above>last) else float(np.max(h))
    return {'support':(support-width,support+width), 'resistance':(resistance-width,resistance+width),
            'atr':atr, 'note':'最近60根日K线的局部转折点+ATR宽度；仅为技术区间启发式估计'}


def evaluate(results: pd.DataFrame, cost_bps: float=15.0) -> dict:
    if results.empty:
        return {}
    r=results.copy()
    mae=float(np.mean(np.abs(r['actual']-r['q50'])))
    coverage=float(np.mean((r['actual']>=r['q10'])&(r['actual']<=r['q90'])))
    direction=float(np.mean((r['q50']>0)==(r['actual']>0)))
    # LONG/CASH only: modeled entry at origin close, exit at fifth future close.
    # This is an IDEALIZED close-to-close proxy, NOT an executable backtest.
    signal=(r['q50']>0).astype(float)
    net=signal*r['actual']-signal*cost_bps/10000
    # Non-overlapping observations, not continuous portfolio daily returns.
    baseline=r['actual']-cost_bps/10000
    targets = {}
    for name in TARGETS:
        prefix = '' if name == 'close' else name + '_'
        if prefix + 'actual' not in r:
            continue
        actual = r[prefix + 'actual'].to_numpy()
        median = r[prefix + 'q50'].to_numpy()
        error = median - actual
        price_error = error * r['price'].to_numpy()
        pinball = {}
        for q in QUANTILES:
            residual = actual - r[prefix + f'q{int(q*100)}'].to_numpy()
            pinball[f'Q{int(q*100)}'] = float(np.mean(np.maximum(q*residual, (q-1)*residual)))
        targets[name] = {
            'MAE': float(np.mean(np.abs(error))),
            'RMSE': float(np.sqrt(np.mean(error**2))),
            'price_MAE': float(np.mean(np.abs(price_error))),
            'price_RMSE': float(np.sqrt(np.mean(price_error**2))),
            'coverage': float(np.mean((actual >= r[prefix+'q10']) & (actual <= r[prefix+'q90']))),
            'direction': float(np.mean((median > 0) == (actual > 0))),
            'pinball': pinball}
    return {'targets': targets, 'MAE':mae,'coverage':coverage,'direction':direction,
            'strategy_compound':float(np.prod(1+net)-1),
            'buyhold_compound':float(np.prod(1+baseline)-1),
            'n':len(r),'strategy_returns':net, 'baseline_returns':baseline}
