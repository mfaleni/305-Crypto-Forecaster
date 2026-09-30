# Old system audit — principal defects

1. `strategy_agent.py` delegates BUY/SELL/HOLD, TP, SL and confidence to an LLM. Confidence is not calibrated from historical outcomes.
2. The strategy prompt consumes heterogeneous snapshots without a tested mapping from inputs to returns.
3. `daily_runner.py` treats Prophet/LSTM outputs as trading inputs without an outcome-validation gate.
4. `data_utils.py` includes placeholder/zero advanced metrics and injects current external snapshots into a historical dataframe, which makes those columns unsuitable as historical predictors.
5. The old horizon is daily / 24–72h, incompatible with the present intraday scalp objective.
6. There is no fee-aware expected-value gate before recommending a trade.
7. There is no complete recommendation -> fill -> exit -> net P&L feedback loop establishing out-of-sample expectancy.

V2 therefore makes the quantitative engine authoritative and removes the LLM from signal creation.
