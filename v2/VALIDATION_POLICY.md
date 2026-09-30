# Validation policy
The product must remain paper/read-only until all of these are satisfied on unseen data and then live paper data:
- At least 200 closed trades across multiple market regimes.
- Positive net expectancy after modeled fees and slippage.
- Profit factor >= 1.20 out of sample.
- Maximum drawdown <= 10% at configured sizing.
- No single walk-forward test window responsible for more than 35% of aggregate profits.
- Paper results for at least 30 calendar days remain directionally consistent with backtest assumptions.
- API key has withdrawals disabled and IP allowlisting is used when supported.
These are engineering gates, not a guarantee of future profitability.
