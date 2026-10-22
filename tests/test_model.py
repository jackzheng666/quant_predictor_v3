import numpy as np
import pandas as pd
from model import make_features, levels, run_research, evaluate


def sample(n=650):
    rng=np.random.default_rng(12)
    c=100*np.exp(np.cumsum(rng.normal(0.0002,0.015,n)))
    d=pd.date_range('2022-01-03',periods=n,freq='B')
    o=c*(1+rng.normal(0,0.004,n))
    h=np.maximum(c,o)*1.007
    l=np.minimum(c,o)*0.993
    return pd.DataFrame({'Date':d,'Open':o,'High':h,'Low':l,'Close':c,'Volume':rng.integers(100000,1000000,n)})


def test_features_label_and_no_future():
    data=sample()
    features=make_features(data)
    assert np.isclose(features.iloc[100].target, data.Close.iloc[105]/data.Close.iloc[100]-1)
    assert features.target.tail(5).isna().all()
    altered=data.copy()
    altered.loc[200:,'Close']*=2
    # Pre-change features strictly causal
    a=make_features(altered)
    np.testing.assert_allclose(features.loc[180:190,'ret5'],a.loc[180:190,'ret5'])


def test_forecast():
    d=sample()
    out=run_research(d,backtest_points=12)
    assert out.low<=out.median<=out.high
    assert not out.results.empty
    assert len(out.results)<=12
    assert (out.results.outcome_date>out.results.Date).all()
    lv=levels(d)
    assert lv['support'][0]<lv['support'][1]
    assert 0<=evaluate(out.results)['coverage']<=1


def test_extreme_targets_exclude_origin_and_require_full_horizon():
    data = sample()
    data.loc[100, 'High'] = 10000
    data.loc[100, 'Low'] = 0.01
    features = make_features(data)
    assert np.isclose(features.loc[100, 'target_highest'], data.High.iloc[101:106].max()/data.Close.iloc[100]-1)
    assert np.isclose(features.loc[100, 'target_lowest'], data.Low.iloc[101:106].min()/data.Close.iloc[100]-1)
    assert features[['target', 'target_highest', 'target_lowest']].tail(5).isna().all().all()


def test_walk_forward_uses_raw_sessions_and_mature_labels(monkeypatch):
    import model
    data = sample(420)
    # Zero-range rows create missing features inside a future test window.
    data.loc[412, ['Open', 'High', 'Low']] = data.loc[412, 'Close']
    calls = []

    def fake_predict(train, test, target='target'):
        origin = int(test.index[0])
        assert (train.index.to_numpy() + model.HORIZON < origin).all()
        expected = make_features(data)
        np.testing.assert_allclose(train[target], expected.loc[train.index, target])
        calls.append((origin, target))
        return np.tile([-0.02, 0.0, 0.02], (len(test), 1))

    monkeypatch.setattr(model, 'fit_predict', fake_predict)
    out = model.run_research(data, backtest_points=3)
    for row in out.results.itertuples():
        origin = int(data.index[data.Date == row.Date][0])
        assert row.outcome_date == data.Date.iloc[origin+5]
        assert row.train_label_end_date < row.Date
        assert np.isclose(row.actual, data.Close.iloc[origin+5]/row.price-1)
    assert set(target for _, target in calls) == set(model.TARGETS.values())
    for target in model.TARGETS:
        assert np.all(np.diff(out.prices(target)) >= 0)
    assert np.all(out.prices('lowest') <= out.prices('close'))
    assert np.all(out.prices('close') <= out.prices('highest'))


def test_future_changes_do_not_change_past_predictions():
    data = sample(420)
    original = run_research(data, backtest_points=2)
    changed = data.copy()
    changed.loc[410:, ['Open', 'High', 'Low', 'Close']] *= 1.5
    later = run_research(changed, backtest_points=2)
    # First origin is 409, so modifications start strictly after its timestamp.
    cols = ['q10', 'q50', 'q90', 'highest_q10', 'highest_q50', 'highest_q90',
            'lowest_q10', 'lowest_q50', 'lowest_q90']
    np.testing.assert_allclose(original.results.iloc[0][cols].astype(float),
                               later.results.iloc[0][cols].astype(float))


def test_all_target_metrics_and_costs():
    results = pd.DataFrame({'price': [100., 200.], 'actual': [.1, -.1],
                            'q10': [-.2, -.2], 'q50': [.05, -.05], 'q90': [.2, .2]})
    for target in ['highest', 'lowest']:
        for column in ['actual', 'q10', 'q50', 'q90']:
            results[target+'_'+column] = results[column]
    metrics = evaluate(results, cost_bps=20)
    for values in metrics['targets'].values():
        assert np.isclose(values['MAE'], .05)
        assert np.isclose(values['price_MAE'], 7.5)
        assert values['coverage'] == 1
        assert all(loss >= 0 for loss in values['pinball'].values())
    np.testing.assert_allclose(metrics['strategy_returns'], [.098, 0])
    np.testing.assert_allclose(metrics['baseline_returns'], [.098, -.102])
