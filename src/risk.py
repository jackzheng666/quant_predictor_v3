"""Conservative position sizing for ordinary main-board shares."""
from decimal import Decimal, ROUND_HALF_UP
import math
from config import DEFAULT


def fee(value, sell=False, cfg=DEFAULT):
    return max(cfg.min_commission, value*cfg.commission) + value*(cfg.transfer_fee + (cfg.sell_tax if sell else 0))


def price_limit(previous, rate=.10):
    def rounded(value):
        return float(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
    return rounded(Decimal(str(previous))*(1-Decimal(str(rate)))), rounded(Decimal(str(previous))*(1+Decimal(str(rate))))


def size_position(price, stop, cash, equity=10_000, stock_value=0, sector_value=0, cfg=DEFAULT):
    if not all(math.isfinite(v) for v in (price, stop, cash, equity, stock_value, sector_value)) or not 0 < stop < price or min(cash, equity, stock_value, sector_value) < 0:
        raise ValueError('Invalid price, stop or account values')
    budget = max(0, min(cash, equity*cfg.max_stock-stock_value, equity*cfg.max_sector-sector_value))
    risk_budget = equity*cfg.risk_per_trade
    execution = price*(1+cfg.slippage)
    shares = int(budget/execution//100)*100
    while shares:
        cost = shares*execution + fee(shares*execution, cfg=cfg)
        exit_value = shares*stop*(1-cfg.slippage)
        loss = cost-exit_value+fee(exit_value, True, cfg)
        if cost <= budget and loss <= risk_budget:
            return dict(shares=shares, cost=cost, estimated_loss=loss, stop=stop)
        shares -= 100
    return dict(shares=0, cost=0., estimated_loss=0., stop=stop)


def stop_level(row):
    return max(.01, min(float(row.support), float(row.Close)-2*float(row.ATR14)))
