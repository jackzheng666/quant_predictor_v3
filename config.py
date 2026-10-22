"""Research assumptions; fees and limits must be checked against broker/security rules."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    capital: float = 10_000
    min_turnover: float = 200_000_000
    max_stock: float = .30
    max_sector: float = .50
    risk_per_trade: float = .01
    commission: float = .0003
    min_commission: float = 5
    sell_tax: float = .0005
    transfer_fee: float = .00001
    slippage: float = .001
    cache_hours: float = 18
    weights: tuple = (25, 25, 20, 15, 15)

DEFAULT = Settings()
