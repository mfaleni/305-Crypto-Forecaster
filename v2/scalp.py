from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ScalpPosition:
    symbol: str
    quantity_sold: float
    exit_price: float
    proceeds_usdc: float
    fee_usdc: float = 0.0

    @property
    def net_proceeds(self) -> float:
        return max(0.0, self.proceeds_usdc - self.fee_usdc)

def rebuy_metrics(position: ScalpPosition, price: float, rebuy_fee_bps: float = 0.0) -> dict:
    if price <= 0:
        raise ValueError("price must be positive")
    fee_rate = rebuy_fee_bps / 10000.0
    spendable = position.net_proceeds * (1.0 - fee_rate)
    qty = spendable / price
    gain = qty - position.quantity_sold
    return {
        "rebuy_price": float(price),
        "rebuy_quantity": float(qty),
        "coin_gain": float(gain),
        "coin_gain_pct": float(gain / position.quantity_sold * 100.0),
        "price_improvement_pct": float((position.exit_price / price - 1.0) * 100.0),
    }

def eth_reentry_plan(position: ScalpPosition) -> list[dict]:
    levels = [
        (2500.0, 0.25, "FIRST"),
        (2425.0, 0.30, "SECOND"),
        (2350.0, 0.45, "HEAVY"),
    ]
    out = []
    for price, fraction, label in levels:
        allocation = position.net_proceeds * fraction
        qty = allocation / price
        out.append({
            "label": label,
            "price": price,
            "fraction": fraction,
            "allocation_usdc": allocation,
            "eth_quantity": qty,
        })
    return out

def btc_reentry_plan() -> list[dict]:
    return [
        {"zone": "$83,500-$84,500", "tranche_usdc": 20000.0, "label": "FIRST"},
        {"zone": "$81,500-$82,500", "tranche_usdc": 25000.0, "label": "SECOND"},
        {"zone": "$78,500-$79,500", "tranche_usdc": 30000.0, "label": "HEAVY"},
        {"zone": "$75,000-$76,500", "tranche_usdc": 27400.0, "label": "MAX VALUE"},
    ]
