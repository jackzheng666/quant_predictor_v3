"""In-memory paper account. No broker integration."""
from dataclasses import dataclass, field
import pandas as pd
from config import DEFAULT
from src.risk import fee, size_position

@dataclass
class Portfolio:
    cash: float = DEFAULT.capital
    positions: dict = field(default_factory=dict)
    ledger: list = field(default_factory=list)

    def equity(self, marks):
        return self.cash + sum(p['shares']*marks.get(c, p['price']) for c,p in self.positions.items())

    def buy(self, code, sector, price, stop, date, marks, cfg=DEFAULT):
        if code in self.positions:
            raise ValueError('Position already held')
        total = self.equity(marks)
        sector_value = sum(p['shares']*marks.get(c,p['price']) for c,p in self.positions.items() if p['sector']==sector)
        order = size_position(price,stop,self.cash,total,sector_value=sector_value,cfg=cfg)
        if not order['shares']:
            return False
        self.cash -= order['cost']
        self.positions[code] = dict(shares=order['shares'], price=price*(1+cfg.slippage),
            stop=stop, date=pd.Timestamp(date), sector=sector, cost=order['cost'])
        self.ledger.append(dict(Date=pd.Timestamp(date), Code=code, Side='BUY', price=price*(1+cfg.slippage),
            fees=fee(order['shares']*price*(1+cfg.slippage),cfg=cfg),
            slippage_cost=order['shares']*price*cfg.slippage, **order))
        return True

    def sell(self, code, price, date, cfg=DEFAULT):
        p = self.positions[code]
        if pd.Timestamp(date) <= p['date']:
            raise ValueError('T+1: cannot sell on purchase date')
        execution = price*(1-cfg.slippage)
        proceeds = p['shares']*execution-fee(p['shares']*execution,True,cfg)
        self.cash += proceeds
        del self.positions[code]
        trade = dict(Date=pd.Timestamp(date), Code=code, Side='SELL', shares=p['shares'],
                     proceeds=proceeds, pnl=proceeds-p['cost'], price=execution,
                     fees=fee(p['shares']*execution,True,cfg), slippage_cost=p['shares']*price*cfg.slippage)
        self.ledger.append(trade)
        return trade
