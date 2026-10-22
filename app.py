"""A-share research dashboard. All live results originate from providers or local cache."""
import datetime as dt
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import DEFAULT
from src.data_fetcher import MarketData, code, peer_benchmarks, normalize
from src.indicators import indicators
from src.screener import screen
from src.risk import size_position
from src.portfolio import Portfolio
from src.backtest import run_backtest, benchmark_curve

st.set_page_config(page_title='Quant Predictor V2',layout='wide')
st.title('Quant Predictor V2 · A股研究工作台')
st.caption('规则筛选 · 3交易日研究周期 · 初始资金 ¥10,000 · 仅回测和模拟交易')
store = MarketData()
with st.sidebar:
    section = st.radio('研究页面',['市场概览','每日 Top 10','个股分析','组合模拟','策略表现'],index=2)
    start = st.date_input('历史开始日期',dt.date.today()-dt.timedelta(days=1100))
    end = st.date_input('结束日期（仅使用昨日及以前日线）',dt.date.today()-dt.timedelta(days=1))
    st.caption('缓存有效期18小时；数据日期以实际返回记录为准。')
    upload = st.file_uploader('可选：自定义股票池 CSV：Code,Name,Sector,Benchmark',type='csv')
    st.caption('无需上传：自动股票池使用真实同行行情构建行业比较基准（至少3只同行）；CSV仅供自定义股票池。')
    if st.button('获取300只主板股票池'):
        try:
            st.session_state.universe = store.universe()
        except Exception as exc:
            st.error(str(exc))
    if 'universe' in st.session_state:
        st.download_button('下载自动股票池',st.session_state.universe.to_csv(index=False).encode('utf-8-sig'),'universe.csv')
    if upload:
        try:
            meta = pd.read_csv(upload,dtype={'Code':str})
            if not {'Code','Name','Sector','Benchmark'} <= set(meta):
                raise ValueError('CSV缺少必要字段')
            meta.Code = meta.Code.map(code)
            if meta.Code.duplicated().any() or meta[['Name','Sector','Benchmark']].isna().any().any():
                raise ValueError('代码重复或元数据存在空值')
            st.session_state.universe = meta.head(300)
        except Exception as exc:
            st.error(str(exc))
    if st.button('自动扫描300只主板股票 / Top 10',type='primary'):
        histories, sectors, failures = {}, {}, []
        meta = st.session_state.get('universe',pd.DataFrame())
        if meta.empty or 'Benchmark' not in meta:
            try:
                with st.spinner('自动获取流动性股票池；备用源可能需数分钟…'):
                    meta = store.universe()
                st.session_state.universe = meta
            except Exception as exc:
                st.error(f'股票池获取失败：{exc}')
        if not meta.empty:
            progress = st.progress(0.)
            for i,row in enumerate(meta.to_dict('records')):
                try:
                    histories[row['Code']] = store.bars(row['Code'],start,end)
                except Exception as exc:
                    failures.append({'Code':row['Code'],'Error':str(exc)})
                progress.progress((i+1)/len(meta))
            sectors.update(peer_benchmarks(histories,meta))
            if 'Benchmark' in meta:
                for name in meta.Benchmark.dropna().unique():
                    if str(name).startswith('peer:'):
                        if name not in sectors:
                            failures.append({'Code':name,'Error':'同行有效历史不足（至少3只），未构造基准'})
                        continue
                    try:
                        sectors[name] = store.sector(name,start,end)
                    except Exception as exc:
                        failures.append({'Code':name,'Error':str(exc)})
            # Modal latest date avoids one stale/suspended symbol shifting all screens.
            latest = [b.Date.iloc[-1] for b in histories.values()]
            asof = pd.Series(latest).mode().max() if latest else pd.Timestamp(end)
            ranked,audit = screen(histories,meta,sectors,asof)
            Path('outputs').mkdir(exist_ok=True)
            ranked.to_csv(f'outputs/{asof:%Y%m%d}_top10.csv',index=False)
            audit.to_csv(f'outputs/{asof:%Y%m%d}_audit.csv',index=False)
            st.session_state.bundle = dict(histories=histories,sectors=sectors,meta=meta,
                asof=asof,ranked=ranked,audit=audit,failures=pd.DataFrame(failures))

bundle = st.session_state.get('bundle')
if section == '个股分析':
    symbol = st.text_input('六位股票代码','600000')
    if st.button('Fetch Market Data',type='primary'):
        st.session_state.pop('stock',None)
        st.session_state.pop('forecast',None)
        st.session_state.pop('stock_errors',None)
        try:
            with st.spinner('下载真实日线及计算技术指标…'):
                raw = store.bars(symbol,start,end)
            st.session_state.stock = (code(symbol),raw,None)
            errors = []
            try:
                adjusted = store.bars(symbol,start,end,'qfq')
                st.session_state.stock = (code(symbol),raw,adjusted)
                if len(adjusted)>=420:
                    from model import run_research
                    with st.spinner('运行原有5交易日分位数研究模型…'):
                        st.session_state.forecast = run_research(adjusted,backtest_points=0)
                else:
                    errors.append('原模型需要至少420个交易日；请将开始日期提前。技术分析仍可使用。')
            except Exception as exc:
                errors.append(f'前复权/模型研究不可用：{exc}；原始行情已保留。')
            st.session_state.stock_errors = errors
        except Exception as exc:
            st.error(f'行情下载失败：{exc}')
    with st.expander('可选：分析自己的原始日线CSV'):
        own = st.file_uploader('原始价格 CSV：Date,Open,High,Low,Close,Volume,Amount',type='csv',key='own_prices')
        st.caption('成交量单位为股，成交额单位为人民币；上传价格为用户提供、未经行情源核验。')
        if own is not None and st.button('使用上传的历史数据'):
            try:
                uploaded = normalize(pd.read_csv(own))
                uploaded['Source'] = 'user-csv-unverified'
                uploaded['Adjustment'] = 'raw-user-declared'
                uploaded['RetrievedAt'] = pd.Timestamp.now(tz='UTC').isoformat()
                st.session_state.stock = (code(symbol),uploaded,None)
                st.session_state.pop('forecast',None)
                st.session_state.stock_errors = ['用户上传数据未经行情源核验，不用于自动股票池筛选。']
            except Exception as exc:
                st.error(f'上传数据无效：{exc}')
    if 'stock' in st.session_state:
        symbol,raw,adjusted = st.session_state.stock
        for message in st.session_state.get('stock_errors',[]):
            st.warning(message)
        options = ['原始可交易价格'] + (['前复权研究价格'] if adjusted is not None else [])
        mode = st.selectbox('价格口径',options)
        selected = raw if mode.startswith('原始') else adjusted
        st.caption(f'{symbol} · 最新交易日 {selected.Date.iloc[-1]:%Y-%m-%d} · 来源 {selected.Source.iloc[-1]} · 复权方法 {selected.Adjustment.iloc[-1]} · 获取时间 {selected.RetrievedAt.iloc[-1]}')
        st.metric('最新核验收盘价',f'¥{selected.Close.iloc[-1]:.2f}')
        d = indicators(selected)
        fig = go.Figure(go.Candlestick(x=d.Date,open=d.Open,high=d.High,low=d.Low,close=d.Close,name='日K'))
        for n in (5,10,20,60):
            fig.add_scatter(x=d.Date,y=d[f'MA{n}'],name=f'MA{n}')
        st.plotly_chart(fig,use_container_width=True)
        st.plotly_chart(go.Figure(go.Bar(x=d.Date,y=d.Volume,name='成交股数')),use_container_width=True)
        for columns,title in [(['RSI'],'RSI'),(['MACD','MACD_signal','MACD_hist'],'MACD'),(['ATR14'],'ATR14')]:
            chart = go.Figure()
            for col in columns:
                chart.add_scatter(x=d.Date,y=d[col],name=col)
            chart.update_layout(title=title,height=220)
            st.plotly_chart(chart,use_container_width=True)
        st.dataframe(d.tail(120),use_container_width=True)
        st.download_button('下载核验行情',d.to_csv(index=False).encode('utf-8-sig'),f'{symbol}.csv')
        if 'forecast' in st.session_state:
            fc = st.session_state.forecast
            st.subheader('原有模型输出 · 5交易日分位数研究')
            st.caption(f'模型日期 {fc.date:%Y-%m-%d} · 来源 {adjusted.Source.iloc[-1]} · qfq研究价格；非3日规则筛选或可执行收益承诺。')
            for col,target in zip(st.columns(3),['close','highest','lowest']):
                prices = fc.prices(target)
                col.metric(target+' Q50',f'¥{prices[1]:.2f}')
                col.caption(f'Q10 {prices[0]:.2f} / Q90 {prices[2]:.2f}')
elif section == '组合模拟':
    st.info('模拟账户由你手动输入价格和日期；记录只保存在当前浏览器会话，不代表市场实际成交。止损可能因T+1或跌停延迟。')
    if 'paper' not in st.session_state:
        st.session_state.paper = Portfolio()
    account = st.session_state.paper
    marks = {c:p['price'] for c,p in account.positions.items()}
    if bundle:
        marks.update({c:float(b.Close.iloc[-1]) for c,b in bundle['histories'].items()})
    st.metric('可用现金',f'¥{account.cash:,.2f}')
    st.metric('模拟净值（可得最新价，缺失时成本价）',f'¥{account.equity(marks):,.2f}')
    c = st.text_input('模拟股票代码','600000')
    sector = st.text_input('行业','银行')
    price = st.number_input('模拟成交参考价',min_value=.01,value=10.)
    stop = st.number_input('计划止损价',min_value=.01,value=9.)
    date = st.date_input('模拟交易日期')
    try:
        sector_value = sum(p['shares']*marks.get(k,p['price']) for k,p in account.positions.items() if p['sector']==sector)
        quote = size_position(price,stop,account.cash,account.equity(marks),sector_value=sector_value)
        st.write('仓位估算（含费用和滑点）',quote)
        if st.button('记录模拟买入'):
            if not account.buy(code(c),sector,price,stop,date,marks):
                st.warning('资金、整手、集中度或风险预算不允许买入')
            else:
                st.rerun()
        if st.button('记录模拟卖出'):
            account.sell(code(c),price,date)
            st.rerun()
    except (ValueError,KeyError) as exc:
        st.error(str(exc))
    st.dataframe(pd.DataFrame(account.positions).T)
    st.dataframe(pd.DataFrame(account.ledger))
elif not bundle:
    st.info('尚未加载市场数据。通过左侧股票池下载历史并筛选；也可以直接进入个股分析或组合模拟。')
    st.caption('进入个股分析点击 Fetch Market Data；模型使用足够长的前复权历史。批量扫描自动准备股票池，无需CSV。')
elif section == '每日 Top 10':
    st.subheader(f"筛选日期 {bundle['asof']:%Y-%m-%d} · {len(bundle['ranked'])} 只合格股票")
    st.caption('自动行业基准为当前股票池同行等权价格收益指数，并非官方行业指数；至少3只同行。催化剂不可用；有效权重：趋势25/85、交易活跃25/85、行业强度20/85、风险15/85。权重为待检验假设。')
    st.dataframe(bundle['ranked'],use_container_width=True)
    st.download_button('下载候选清单',bundle['ranked'].to_csv(index=False).encode('utf-8-sig'),'top10.csv')
    st.caption('EntryTrigger 为突破当日最高价的观察价；尚未实现条件单成交模型。原始价格跨除权可能影响技术指标。')
    st.dataframe(bundle['audit'],use_container_width=True)
elif section == '市场概览':
    st.caption(f"样本股票池概览 · 日期 {bundle['asof']:%Y-%m-%d}；不是全市场指数。")
    observations = []
    for c,b in bundle['histories'].items():
        if len(b)>1 and b.Date.iloc[-1] == bundle['asof']:
            observations.append({'Code':c,'Return':b.Close.iloc[-1]/b.Close.iloc[-2]-1,'Amount':b.Amount.iloc[-1]})
    obs = pd.DataFrame(observations)
    if not obs.empty:
        st.metric('上涨比例（样本）',f'{(obs.Return>0).mean():.1%}')
        st.metric('成交额（样本）',f'¥{obs.Amount.sum()/1e8:,.1f}亿')
        st.dataframe(obs)
    sector_rows = []
    for name,b in bundle['sectors'].items():
        if len(b)>=6:
            sector_rows.append({'Sector':name,'Date':b.Date.iloc[-1],'Return5':b.Close.iloc[-1]/b.Close.iloc[-6]-1})
    st.write('行业5日强度（已加载板块）')
    st.dataframe(pd.DataFrame(sector_rows))
    st.write('数据获取失败')
    st.dataframe(bundle['failures'])
elif section == '策略表现':
    st.warning('使用当前股票池/行业标签存在幸存者偏差。此回放不代表严格点时研究。未实现分红、配股及退市清算；检测价格断点时绩效无效。')
    hold = st.selectbox('持有交易日数',[1,3,5],index=1)
    names = list(bundle['sectors'])
    if not names:
        st.info('需加载完整行业基准，才能构建独立交易日历及基准比较。')
    else:
        bench = st.selectbox('价格收益比较基准（官方板块或同行代理）',names)
        split = st.date_input('测试集开始日期（之前数据仅作指标预热）',dt.date.today()-dt.timedelta(days=180))
        if st.button('运行历史回放'):
            def signals(date,hist):
                result,_ = screen(hist,bundle['meta'],bundle['sectors'],date)
                return result
            try:
                calendar = store.trading_dates(split,bundle['asof'])
                result = run_backtest(bundle['histories'],calendar,signals,hold)
                result['equity']['Benchmark'] = benchmark_curve(bundle['sectors'][bench],calendar)
                st.session_state.bt = result
            except Exception as exc:
                st.error(str(exc))
        if 'bt' in st.session_state:
            result = st.session_state.bt
            if not result['metrics']['valid']:
                st.error('价格断点导致绩效无效；请勿据此评价策略。')
            st.json(result['metrics'])
            st.line_chart(result['equity'])
            st.dataframe(result['trades'])
            st.dataframe(result['blocked'])
            st.dataframe(result['actions'])
            st.download_button('下载净值',result['equity'].to_csv().encode('utf-8-sig'),'equity.csv')
