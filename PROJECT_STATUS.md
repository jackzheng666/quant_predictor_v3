# Project status — 2026-10-14 (Asia/Shanghai)

## Implemented

- Inspected all original source, dependencies and tests before editing. Repository has no initial commit and contains pre-existing staged files. Source backup created at `backups/original-source.tar.gz`; staged state intentionally preserved.
- Preserved `data.py`, `model.py`, existing tests and previous app as `legacy_app.py`; redirected the original app integration test to that preserved UI.
- Added configuration, primary AKShare / backup BaoStock daily adapters, standardized codes, OHLC/amount validation, units, metadata, and persistent SQLite exact-range cache.
- Separate raw execution prices and qfq research chart series; no fake market data. Exclude today’s partial bar. API errors are raised; fallback warnings identify the original failure.
- Causal indicators, strict rules, missing-factor weight normalization, Top 10 plus exclusion audit. Automatic BaoStock industry membership and real sector-peer benchmarks remove the upload requirement; optional official board mapping remains supported. Generated CSVs persist under ignored `outputs/`.
- 100-share lot sizing with cash, stock/sector concentration, ATR/support stop, risk budget, fees and slippage. Manual session-local paper account with T+1 guard.
- Next-open event simulation using an independent trading calendar, 1/3/5-session holding periods, suspension/limit checks, delayed exits, gross/net curves, cost totals, trade ledger, drawdown, win rate, profit factor, benchmark price-return comparison, test-start warm-up and chronological split/embargo helper.
- Streamlit sections for sample-market overview, daily candidates, individual raw/qfq candles and indicators, manual portfolio and historical strategy replay.
- Existing `.venv` already has all required packages including BaoStock; no installation required.

## Verification

- ` .venv/bin/python -m pytest -q`: 22 tests passed in 12.97 seconds, no warnings in the final run.
- ` .venv/bin/python -m compileall -q app.py config.py src`: passed.
- ` .venv/bin/python -m pip check`: no broken requirements.
- Verified actual AKShare raw prices for `600000`, requested 2026-09-01 through 2026-10-12. Latest returned close: 9.67 on 2026-10-12; cache records source/adjustment/retrieval metadata. Initial sandbox DNS failure was reported; approved network retry succeeded.
- Tests use explicitly synthetic fixtures/mocked providers only for calculation and UI validation. They do not establish investment returns. Original fallback mechanics used mocked tests; subsequent live BaoStock history, industry metadata, automatic universe and exchange-calendar checks succeeded (see automatic retrieval update).

## Limitations / next priorities

1. **Point-in-time universe and sector membership unavailable.** Current 300-stock turnover snapshot and current industry mapping introduce survivorship/membership bias in historical replay. Automatic peer benchmarks require at least three observed peers and are explicitly labeled proxies, not official sector indices; optional official-board names must match the provider.
2. **Corporate actions not accounted for.** Raw-price technical returns may jump at ex dates. Detected previous-close/reference discontinuities freeze fills and invalidate performance. Dividends, splits, rights and delisting settlement require explicit ledgers. qfq is a separately labeled provider-revised research series.
3. Daily price-limit checks use ordinary seasoned non-ST main-board 10% assumptions and available reference prices; exceptional securities/rules need authoritative effective-dated metadata. No queue-priority or intraday fill model. Stops are close-monitored/next-open, not guaranteed stop orders.
4. The manual portfolio uses user-input hypothetical prices/dates, costs and marks. T+1 disallows same-calendar-date sale, but without an authoritative exchange calendar it does not validate all holidays; actual backtest uses independent benchmark trading dates. Paper account and loaded UI state are session-local, not durable account records.
5. UI supports chronological test-start/warm-up, but no parameter optimization or ML training in the new engine. Existing legacy ML remains separate and its historical close-to-close proxy is unsuitable as executable performance.
6. Fee estimates need broker-specific validation. Benchmark is an industry price-return series without costs/dividends, not a matched tradable portfolio. Final open positions remain marked rather than forcibly liquidated.
7. Exact-range cache is valid for 18h; incremental overlapping refresh, bulk-provider throttling/retries and persistent point-in-time snapshot version history remain future work. Today is excluded even after close.
8. There are no verified catalyst/news factors, calibrated probabilities or profitability claims. Next: effective-dated universe/status/calendar/action data, then validated walk-forward 3-day relative-return Logistic Regression/XGBoost and source-backed news/announcement analysis.
9. `.gitignore` excludes generated data, bytecode, credentials and `.venv`. Some bytecode was already staged by the user before this session; clean those index entries before an initial commit. No commit, push or live order performed.

## Resume

Read `README.md`, `config.py`, this file, and run the tests. Launch with `.venv/bin/python -m streamlit run app.py`; the old UI is `legacy_app.py`. Keep data omissions honest and do not replace unavailable factors or results with examples.

## Automatic retrieval update — 2026-10-14

- Default landing page is individual analysis with **Fetch Market Data**. No CSV needed. Added Plotly RSI/MACD/ATR plots and original `model.run_research` outputs with separate qfq provenance. Default history enlarged to supply enough observations. Model/adjusted failures preserve raw charts; failed new symbols clear previous results.
- Automated 300-name universe and scan, AKShare primary, real BaoStock HS300+ZZ500 turnover-ranked main-board pool backup. Industry names come from BaoStock. Actual peer returns produce labeled sector proxies; minimum-three-peer and missing-history exclusions apply. Rule weights, fees, lots, concentration and stops remain unchanged.
- Independent BaoStock trading calendar added for replay. Optional custom universe and personal OHLCV/amount CSV retained; uploads labeled unverified.
- Live verification: `600000`, 2023-01-01 to 2026-10-13, **912 raw and 912 qfq bars** via BaoStock after AKShare network failure. Existing model produced an actual 2026-10-13 forecast; no market data or probabilities fabricated. Live BaoStock industry membership also verified.
- New regression coverage: provider universe fallback, causal peer benchmarks and gaps, raw-chart survival after qfq failure, updated automatic UI workflow. Full 300-stock live scan is not a validated strategy-performance result.

- Updated regression run: **26 tests passed in 12.97s**; compile, whitespace and dependency checks passed. Real cached-data Streamlit run: no exceptions/errors, five Plotly charts, actual model quantile outputs. Existing local service responds HTTP 200 at http://127.0.0.1:8501.

- Live automatic universe validation completed: **300 real main-board stocks** with BaoStock industry/company metadata, ranked by observed latest turnover within HS300+ZZ500 backup constituents. Independent trading dates verified for 2026-10-01 to 2026-10-13. The complete 300-symbol historical scan has not been executed during this session; unit/UI coverage and live universe retrieval are distinct checks.

## Import/startup investigation — 2026-10-14

`peer_benchmarks` remains a real top-level function in `src/data_fetcher.py`; `MarketData`, `code`, and `normalize` also import successfully. All application modules import in a fresh virtual-environment process, with no circular import or signature mismatch identified. Two old Streamlit instances from this workspace were listening on IPv4/IPv6 port 8501; stale process state is the suspected cause of the reported ImportError, which could not be reproduced in a fresh process. Added `run_app.sh` to use the absolute project entry point and existing interpreter, with polling for source changes; no strategy or benchmark calculation changed.

Verification after restart: all nine core modules imported successfully; **26 tests passed in 13.15s**. Fresh Streamlit AppTest rendered Fetch Market Data with zero exceptions. One new server listens on 127.0.0.1:8501; health check returned ok and homepage HTTP 200. Modified files for this fix: `run_app.sh`, `README.md`, `PROJECT_STATUS.md`. Existing data-fetcher and screening implementation preserved.
