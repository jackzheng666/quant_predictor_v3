import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data import get_prices
from model import run_research, levels, evaluate

st.set_page_config(page_title='Quant Price Predictor V2', page_icon='📈', layout='wide')
st.title('📈 Quant Price Predictor V2')
st.caption('A股 / 美股 · 5交易日预测 · LightGBM分位数回归 · 时间顺序样本外检验 · 仅供研究')
st.warning('预测区间不是保证区间；支撑/压力是历史技术区间，并非机器学习价格预测。回测为收盘价成交的理想化代理，不能直接视为可交易收益。')
with st.sidebar:
    st.header('研究参数')
    market=st.selectbox('市场',['A股','美股'],index=0)
    ticker=st.text_input('股票代码','300308' if market=='A股' else 'NVDA').strip().upper()
    start=st.date_input('开始日期',dt.date(2019,1,1),max_value=dt.date.today())
    points=st.slider('样本外回测期数（每期5日）',12,48,24,step=6)
    cost=st.number_input('往返交易成本（基点 bps）',min_value=0.0,max_value=200.0,value=15.0,step=5.0)
    go_btn=st.button('下载行情并运行研究',type='primary',use_container_width=True)
    st.caption('首次运行可能需等待数十秒至数分钟，取决于网络和CPU。')

@st.cache_data(ttl=3600,show_spinner=False)
def load(t,m,s,e):
    return get_prices(t,m,s,e)

if go_btn:
    try:
        with st.spinner('下载行情、训练分位数模型、进行滚动回测…'):
            bars=load(ticker,market,start,dt.date.today())
            forecast=run_research(bars,backtest_points=points)
            lv=levels(bars)
            metrics=evaluate(forecast.results,cost)
        st.session_state['out']=(ticker,market,bars,forecast,lv,metrics,cost)
    except Exception as e:
        st.error(f'运行失败：{e}')
        st.info('请确认联网、代码正确并确保选取足够长的历史数据。A股接口在海外网络环境可能偶尔不可用。')

if 'out' not in st.session_state:
    st.info('从左侧输入代码，点击「下载行情并运行研究」。例如 A股 300308、002594；美股 NVDA、TSLA、MSFT。')
    st.stop()

ticker,market,bars,fc,lv,metrics,cost=st.session_state['out']
st.subheader(f'{ticker} | 截至 {fc.date:%Y-%m-%d} 收盘后的研究预测')
st.caption(f'所用历史行情最后一个交易日：{bars.iloc[-1]["Date"]:%Y-%m-%d} · 币种：{"人民币" if market=="A股" else "美元"} · 预测第5个交易日收盘价，以及未来第1–5个交易日的最高/最低价。')
st.metric('最近收盘（复权序列）', f'{fc.close:.2f}')
labels = {'close': '第5日收盘', 'highest': '未来5日最高', 'lowest': '未来5日最低'}
for column, (target, label) in zip(st.columns(3), labels.items()):
    prices = fc.prices(target)
    column.metric(label + ' Q50', f'{prices[1]:.2f}', f'{fc.quantiles[target][1]:+.2%}')
    column.caption(f'Q10: {prices[0]:.2f} · Q90: {prices[2]:.2f}')
st.caption('各目标分别训练Q10/Q50/Q90；10%–90%区间名义覆盖率80%。最高/最低价是整个未来5交易日的极值，不是每日价格路径。')
left,right=st.columns(2)
left.info(f'历史支撑参考区间：{lv["support"][0]:.2f} – {lv["support"][1]:.2f}')
right.info(f'历史压力参考区间：{lv["resistance"][0]:.2f} – {lv["resistance"][1]:.2f}')
st.caption(lv['note']+'。使用了复权OHLC，可能与实时交易报价不同，尤其在公司行为发生之后。')
plot_bars=bars.tail(130)
fig=go.Figure(data=[go.Candlestick(x=plot_bars['Date'],open=plot_bars['Open'],high=plot_bars['High'],
                                    low=plot_bars['Low'],close=plot_bars['Close'],name='日K')])
for label,bounds,color in [('支撑',lv['support'],'green'),('压力',lv['resistance'],'red')]:
    fig.add_hrect(y0=bounds[0],y1=bounds[1],fillcolor=color,opacity=.12,line_width=0,
                  annotation_text=label,annotation_position='top left')
# Place forecasts in a categorical panel: no invented exchange holiday dates.
for target, color in [('close', '#4477AA'), ('highest', '#CC6677'), ('lowest', '#228833')]:
    prices = fc.prices(target)
    fig.add_hline(y=float(prices[1]), line_dash='dot', line_color=color,
                  annotation_text=labels[target] + ' Q50', annotation_position='bottom right')
fig.update_layout(title='最近130日复权K线 + 预测中位价格 + 历史技术区间',height=450,xaxis_rangeslider_visible=False)
st.plotly_chart(fig,use_container_width=True)
forecast_chart = go.Figure()
for target, color in [('close', '#4477AA'), ('highest', '#CC6677'), ('lowest', '#228833')]:
    prices = fc.prices(target)
    forecast_chart.add_trace(go.Scatter(
        x=[labels[target]], y=[prices[1]], name=labels[target], mode='markers',
        marker=dict(size=12, color=color),
        error_y=dict(type='data', symmetric=False, array=[prices[2]-prices[1]],
                     arrayminus=[prices[1]-prices[0]])))
forecast_chart.update_layout(title='未来5交易日预测：Q50及Q10–Q90区间', yaxis_title='复权价格', height=350)
st.plotly_chart(forecast_chart, use_container_width=True)
st.subheader('样本外滚动验证（5日不重叠的测试窗口）')
if not metrics:
    st.warning('样本外窗口不足，尝试延长数据时间范围。')
else:
    x1,x2,x3,x4=st.columns(4)
    x1.metric('方向准确率',f'{metrics["direction"]:.1%}')
    x2.metric('分位区间覆盖率',f'{metrics["coverage"]:.1%}')
    x3.metric('收益率预测MAE',f'{metrics["MAE"]:.2%}')
    x4.metric('测试窗口数量',str(metrics['n']))
    accuracy = pd.DataFrame([
        {'目标': labels[target], '价格MAE': values['price_MAE'], '价格RMSE': values['price_RMSE'],
         '收益率MAE': values['MAE'], '区间覆盖率': values['coverage'],
         **{f'{q} pinball': loss for q, loss in values['pinball'].items()}}
        for target, values in metrics['targets'].items()])
    st.dataframe(accuracy, use_container_width=True)
    res=fc.results.copy()
    for target, label in labels.items():
        prefix = '' if target == 'close' else target + '_'
        actual = res['price'] * (1 + res[prefix+'actual'])
        median = res['price'] * (1 + res[prefix+'q50'])
        accuracy_chart = go.Figure()
        accuracy_chart.add_trace(go.Scatter(x=res['Date'], y=res['price']*(1+res[prefix+'q90']),
                                           mode='lines', line=dict(width=0), showlegend=False))
        accuracy_chart.add_trace(go.Scatter(x=res['Date'], y=res['price']*(1+res[prefix+'q10']),
                                           fill='tonexty', mode='lines', line=dict(width=0), name='Q10–Q90'))
        accuracy_chart.add_trace(go.Scatter(x=res['Date'], y=median, name='预测 Q50'))
        accuracy_chart.add_trace(go.Scatter(x=res['Date'], y=actual, name='实际'))
        accuracy_chart.update_layout(title=label + '：样本外预测与实际', xaxis_title='预测日',
                                     yaxis_title='复权价格', height=320)
        st.plotly_chart(accuracy_chart, use_container_width=True)
    res['strategy_curve']=(1+metrics['strategy_returns']).cumprod()-1
    res['baseline_curve']=(1+metrics['baseline_returns']).cumprod()-1
    chart=go.Figure()
    chart.add_trace(go.Scatter(x=res['outcome_date'],y=res['strategy_curve'],name='模型择时: 做多/空仓'))
    chart.add_trace(go.Scatter(x=res['outcome_date'],y=res['baseline_curve'],name='始终持有: 5日窗口基准'))
    chart.update_layout(title=f'理想化5日窗口累计收益（每次进出总成本 {cost:g} bps）',height=400,
                        yaxis_tickformat='.0%')
    st.plotly_chart(chart,use_container_width=True)
    st.caption('此图假设在信号当天收盘价买入、第5个交易日收盘价卖出；这不是实际可保证的成交价。未建模A股100股整手、涨跌停、停牌、流动性、税费明细及跨日资金约束。未来5日收益以日线复权价格估计，不等于真实账户净值。')
    table=res.drop(columns=['strategy_curve','baseline_curve']).copy()
    st.dataframe(table.rename(columns={'Date':'预测日','outcome_date':'结果日','price':'基准价','q10':'10%分位收益','q50':'中位收益','q90':'90%分位收益','actual':'实际5日收益'}),use_container_width=True)
    st.download_button('下载样本外预测CSV',res.to_csv(index=False).encode('utf-8-sig'),file_name=f'{ticker}_backtest.csv',mime='text/csv')

with st.expander('模型说明与重要限制'):
    st.markdown('''- 输入：仅使用预测日及之前的日线价格、成交量、波动率、均线与RSI。
- 目标：未来第5个交易日收盘、未来1–5日最高价和最低价相对当前收盘的收益率；每个目标分别训练10%、50%、90%分位数 LightGBM（共9个模型）。
- 验证：逐个向前滚动，测试时间点相隔5个交易日；训练前剔除最近5日未完全成熟的标签，避免未来信息泄漏。
- 技术区间：使用60根K线局部高低点与14日ATR估计；不是模型预测的到价概率。
- 缺陷：数据源可能修改历史前复权因子，存在公司行为的历史重算偏差；没有点时历史数据快照。使用当前复权历史，故**回测并非严格点时无偏**。
- 市场：不同股票须分别训练，不会自动推断跨市场因果关系；美股与A股按各自交易日计5个交易日。
- 预测日：页面显示的是最后一个下载到的交易日，不使用实时盘中数据。''')
