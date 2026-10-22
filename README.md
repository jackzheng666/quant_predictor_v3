# Quant Predictor V2

A-share main-board research workbench: provider-backed daily data, causal rule screening, conservative sizing, paper account and next-open historical replay. No real orders are sent. The old single-stock A-share/US LightGBM research code is preserved in `data.py`, `model.py`, and `legacy_app.py`.

## Run on macOS / Python 3.13

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
./run_app.sh
```

Open http://localhost:8501. Run the previous research UI with `python -m streamlit run legacy_app.py`. Its close-to-close ML evaluation remains an idealized proxy and is separate from the new event simulation.

```bash
python -m pytest -q
python -m pip check
```

## Workflow

1. The dashboard opens **Individual Stock Analysis** by default. Enter `600000`, `000001` or `600519` and click **Fetch Market Data**. No CSV is required. Raw and qfq daily prices use AKShare first, BaoStock second, with an 18-hour SQLite cache. Default history covers ~3 years so the original model can run when sufficient data exists.
2. Inspect candles, volume, MA5/10/20/60, RSI, MACD and ATR14. Every price panel labels the actual latest date, provider and adjustment. If qfq/model processing fails, verified raw data remains visible. The original 5-session LightGBM outputs are integrated without changing the rule strategy; short histories honestly show model unavailability.
3. Click **自动扫描300只主板股票 / Top 10** to fetch the universe and histories automatically. AKShare turnover-ranked main-board stocks are primary. If unavailable, BaoStock ranks real latest turnover in the combined current HS300+ZZ500 main-board constituent pool. The backup scope is narrower than the entire market and may yield fewer than 300 eligible names; refreshing it can take minutes.
4. Automatic sector assignment uses actual BaoStock industry metadata. The default benchmark is a clearly labeled **equal-weight daily price-return index derived from available real same-industry peers in the downloaded pool**, requiring at least three observed peers per date. It is not an official sector index. Missing history is not invented, unknown industry or insufficient peers still excludes a stock. Current membership and self-inclusion create historical selection/comparison bias; rule conditions and scoring weights are unchanged.
5. CSV uploads are optional: custom universe `Code,Name,Sector,Benchmark` (official Eastmoney board names or `peer:` industry keys); optional raw personal-price CSV with `Date,Open,High,Low,Close,Volume,Amount` in the individual-analysis expander. User uploads are labeled unverified and do not enter automatic screening.
6. Use portfolio and historical replay as before. Backtesting obtains an independent BaoStock exchange calendar; missing comparison benchmark dates cause an explicit error rather than synthetic values. The original UI remains available at `legacy_app.py`.

## Rules and assumptions

`config.py` holds capital (¥10,000), prior-20-session average turnover (>¥200m), concentration caps (30% stock, 50% sector), risk budget (1% equity), fee and slippage assumptions. At ¥10,000, even one lot can be unaffordable under these caps. A zero-share recommendation is valid.

Screen: ≥60 observed trading bars, MA5 > MA10 > MA20, positive 5-session momentum, turnover above the **preceding** 20-session average, positive 5-session relative return versus the mapped sector, positive volume and no current price-limit close. Includes RSI, MACD, ATR14, prior-20-session breakout, annualized volatility, support/resistance. Research weights are 25/25/20/15/15 for trend/activity/sector/catalyst/risk. Catalyst data is unavailable: its weight is excluded and the other weights are divided by 85. No predictions or probabilities are fabricated.

The screen uses raw prices so support, triggers and stops refer to executable price units. The individual analysis separately displays qfq research prices. Raw historical returns/indicators can be distorted across corporate actions. qfq can be revised by providers and is not a point-in-time snapshot. Neither is silently mixed with raw execution prices.

Fees default to 0.03% commission with ¥5 minimum per side, 0.001% transfer fee, 0.05% sell tax, and 0.1% slippage per side. These are **configurable estimates**, not a verified broker quotation. Position loss includes estimated entry/exit fees and slippage; stop loss is not a bound on actual loss. Gap/limit/T+1 conditions can cause larger losses.

Ordinary seasoned non-ST main-board stocks use a 10% limit rounded to cents (half up), with provider `PreClose` or reference reconstructed from the provider's rounded percentage change. Securities with exceptional limits require authoritative per-date `LimitRate`/reference metadata. ST, new listings and other boards are outside the implemented trading model.

## Data/cache

AKShare `stock_zh_a_hist` is primary; BaoStock daily history is backup. Eastmoney volume is converted from 100-share lots to shares; turnover is RMB. BaoStock volume is already shares. Normalize numeric fields, reject impossible OHLC/negative quantities, warn about rejected rows, deduplicate dates and retain zero-volume suspension records. No missing bars are manufactured.

SQLite `data/market.sqlite` stores serialized snapshots keyed by symbol, requested date interval and adjustment, plus retrieval timestamps. Exact repeated requests reuse a cache for 18 hours; overlapping ranges are not yet incrementally merged. Expired-cache failures are reported rather than silently using stale data. Provider errors are surfaced; fallback success includes a warning and records the actual source. Today is excluded even after close to avoid partial-day daily bars. Non-trading days remain absent. Network access is necessary to refresh data; provider availability and sector endpoints may vary.

## Backtest semantics

`src/backtest.py` accepts raw histories, an independent benchmark/exchange calendar and a signal callback receiving only histories up to the signal date. The dashboard reuses the rule screen and independently downloads BaoStock exchange trading dates. Signals execute only at the **next session's open**, expire after that session if unfillable, and respect 100-share lots, cash, concentration, fees, slippage and T+1. Entry triggers shown in the candidate table are not used as stop-buy orders in this baseline next-open model.

Holding periods are 1, 3 or 5 exchange sessions after purchase. Positions are not sold on purchase day. A stop is monitored at previous available close, then exited at the next eligible open; intraday stop fills are not modeled. Suspensions, missing references, and limit opens defer exits; opening-price buys at the upper limit and sells at the lower limit are rejected conservatively. Daily OHLC cannot establish actual queue priority or intraday execution, so feasible modeled fills are not guaranteed. Open positions are marked using the latest observed close, not forcibly liquidated at the backtest end.

Gross and net curves use the **same trades and share quantities**; gross removes fees/slippage from cashflows while sizing remains based on the net account. Reports include returns, win rate, drawdown, profit factor (unavailable without realized losses), fee totals, slippage and blocked orders. The benchmark is a clearly labeled industry **price-return** curve without trading costs, not a comparable invested account or dividend total return.

Corporate-action discontinuities between previous close and the next bar's reference freeze future fills in that security and invalidate the run's performance. Actual splits/dividends/rights/terminal delisting cashflows are not implemented. The displayed invalid curve is diagnostic only.

The UI supports a test-start date with preceding history retained for indicator warm-up. `chronological_split` provides chronological train/test separation with an embargo for future ML labels. Current membership/industry assignments are used historically: **survivorship and membership bias remain**. No profitability claim is made. For defensible evaluation, acquire point-in-time universes including delisted stocks, effective-dated ST/limit status, corporate-action ledgers, and benchmark total returns.

## Modules

- `app.py`: five-section Streamlit dashboard.
- `config.py`: research/risk/cost settings.
- `src/data_fetcher.py`: provider adapters, metadata, sector data, SQLite cache.
- `src/indicators.py`: chronological technical features.
- `src/screener.py`: strict rules, scores, Top 10 and audit.
- `src/risk.py`: lots, fee estimates, limit rounding and sizing.
- `src/portfolio.py`: session-local paper account, ledger and T+1 guard.
- `src/backtest.py`: next-open simulation, benchmark and temporal split.
- `tests/`: synthetic mechanics tests, mocked providers, and Streamlit integration tests.
- `PROJECT_STATUS.md`: verified state and remaining work.

Ignored: virtual environments, caches, backups, output files, datasets and credentials. The initial repository already staged some bytecode; those index entries are left untouched. Remove them from the index before your first commit. Original source backup: `backups/original-source.tar.gz`.

Future work: point-in-time data and corporate-action accounting first; then walk-forward Logistic Regression/XGBoost for 3-day relative returns, probability calibration, verified announcement/news summarization and provenance-aware catalysts. Existing ML is optional legacy research and does not feed current rankings.

Reference for exchange-rule review: [SSE Trading Rules, 2026 revision](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml), [BaoStock](https://www.baostock.com/). Verify security-specific effective rules before extending this simulator.

If an import succeeds in a fresh Python process but fails in an already-running dashboard, stop the old Streamlit process before restarting. `run_app.sh` fixes the workspace/interpreter and enables polling for source changes. Avoid running multiple servers on IPv4/IPv6 port 8501.
