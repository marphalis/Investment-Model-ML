# Investment Model ML — USO

A machine-learning project exploring whether hourly market data for the United States Oil Fund (USO) can be used to generate directional trading signals.

> **Current status:** Initial data preparation, feature engineering and classification experiments are complete. A transaction-cost-aware backtest and robustness analysis remain to be implemented. No profitability claim is made at this stage.

## Project objective

The objective is to evaluate whether models can predict the direction of the next observed USO closing price and whether their predictions can be translated into a trading strategy.

The current classification target is `1` when the next observed close is higher than the current close, and `0` otherwise. The horizon is one observation, not necessarily one clock hour, because the dataset contains irregular time gaps.

## Data

The project currently focuses on `USO.csv` and its cleaned version, `USO_clean.csv`. The data contains hourly OHLCV observations: timestamp, open, high, low, close and volume.

The broader assignment also provides `SQQQ.csv` and `BRENT.csv`; they are not included in the results below.

## Feature engineering

| Feature | Definition | Interpretation |
|---|---|---|
| `time_gap` | Difference between consecutive timestamps, in hours | Time elapsed since the previous observation |
| `internal_return` | `(close - open) / open` | Return within the current candle |
| `cc_return` | `close.pct_change()` | Return from the previous observed close |
| `hour` | Hour extracted from timestamp | Intraday time-of-day information |
| `hourly_relative_volume` | Current volume divided by the rolling mean of previous volumes for the same clock hour | Activity relative to usual volume at that hour |
| `relative_volume` | Current volume divided by the rolling mean of previous observations' volume | Activity relative to recent volume |
| `volatility` | Rolling standard deviation of `cc_return` | Recent variability of close-to-close returns |

The relative-volume calculations use `shift(1)`, excluding the current observation from the historical volume baseline. Rolling windows are measured in observations, not fixed clock-time intervals.

Features based on the current candle are only available after that candle has closed. The trading simulation must therefore execute after the signal is generated, not retrospectively at the same candle's close.

## Target and validation design

The current target uses the next observed close:

```python
future_close = data["close"].shift(-1)
data["target_1"] = np.where(
    future_close.notna(),
    (future_close > data["close"]).astype(int),
    np.nan
)
```

The data is divided chronologically into an 80% training period and a 20% test period. The final training observation is purged because its target would otherwise depend on a price in the test period.

Hyperparameter selection uses five-fold `TimeSeriesSplit` cross-validation with `gap=1`. The test period is kept separate from model selection.

## Models and classification results

The following models were evaluated: Decision Tree tuned by `ccp_alpha`, Decision Tree tuned by `max_leaf_nodes`, and Random Forest. Hyperparameters were selected using `GridSearchCV` and `balanced_accuracy`.

| Model | Best parameters | Best CV balanced accuracy | Test accuracy | Test balanced accuracy |
|---|---|---:|---:|---:|
| Decision Tree (`ccp_alpha`) | `ccp_alpha=0.00032017` | 0.5177 | 0.5165 | 0.5184 |
| Decision Tree (`max_leaf_nodes`) | `max_leaf_nodes=7` | 0.5262 | 0.5232 | 0.5239 |
| Random Forest | `n_estimators=200`, `max_depth=3`, `min_samples_leaf=10`, `max_features=sqrt` | 0.5313 | 0.5211 | 0.5216 |

The majority-class baseline achieved 0.4928 test accuracy and 0.5000 balanced accuracy.

The `max_leaf_nodes` Decision Tree achieved the highest test accuracy and balanced accuracy among these models. The Random Forest achieved the highest cross-validation balanced accuracy. The differences are modest; classification results alone do not demonstrate a profitable trading strategy.

## Trading strategy and backtesting

### Initial strategy definition

The initial strategy is **Long/Hold**:

- Prediction `1`: hold a long USO position.
- Prediction `0`: hold cash.
- Maximum exposure: 100% of available capital.
- No leverage and no short selling.

This simple strategy tests whether the classifier adds value over remaining invested or staying in cash. A short-selling strategy can be assessed separately.

### Execution timing and target alignment

A prediction made after candle *t* closes cannot realistically be executed at that same close without an explicit execution assumption.

A realistic simulation should:

1. Wait for candle *t* to close.
2. Calculate features and generate the prediction.
3. Execute at the next available price, initially approximated by the open of candle *t+1*.
4. Hold until the next scheduled rebalance, initially the open of candle *t+2*.
5. Calculate the realised return from those executable prices.

**Important target-alignment limitation:** the current classifier predicts whether `close[t+1] > close[t]`, whereas the proposed executable holding-period return is based on `open[t+2] / open[t+1] - 1`. These are different price intervals. For a rigorous strategy evaluation, retrain the model with a target aligned to the intended execution interval:

```python
# Signal generated after close[t].
# Entry at open[t+1], exit at open[t+2].
future_execution_return = (
    data["open"].shift(-2) / data["open"].shift(-1) - 1
)

data["execution_target"] = (
    future_execution_return > 0
).astype("float")
data.loc[future_execution_return.isna(), "execution_target"] = np.nan
```

Features in row *t* must still contain only information available by close *t*. Exclude observations without a complete future execution interval. Retrain and reevaluate the models using this aligned target before interpreting the backtest as evidence of trading value.

### Return and transaction-cost calculation

Define:

- `position[t]`: 1 for a long position, 0 for cash.
- `gross_return[t]`: return from executable entry to executable exit.
- `turnover[t]`: absolute change in position.
- `cost[t]`: transaction cost on traded notional.
- `net_return[t]`: gross return less transaction costs.

The following is a **mechanics template**, not a completed backtest. `predictions` must be out-of-sample predictions, and the indices must be aligned to the signal timestamps.

```python
# Each prediction is made after close[t].
# Entry: open[t+1]; exit: open[t+2].
position = pd.Series(predictions, index=X_test.index).astype(float)

entry = data.loc[position.index, "open"].shift(-1)
exit_ = data.loc[position.index, "open"].shift(-2)
gross_return = exit_ / entry - 1

# Illustrative assumption only; replace with a justified cost estimate.
one_way_cost = 0.001

turnover = position.diff().abs()
turnover.iloc[0] = position.iloc[0]  # initial entry from cash

transaction_cost = turnover * one_way_cost
strategy_return = position * gross_return - transaction_cost

strategy_return = strategy_return.dropna()
equity_curve = (1 + strategy_return).cumprod()
```

The exact index alignment should be checked carefully when implementing the execution target and predictions. Do not use in-sample predictions to report test-period performance.

The example cost of `0.001` is illustrative, not an estimate of actual USO trading costs. The final backtest should account for both sides of a trade. If bid/ask quotes are available, use them directly; otherwise document spread and commission assumptions and apply them consistently. Include slippage if a defensible estimate is available.

### Required backtest metrics

Report performance after costs, and gross performance where useful for comparison:

- Cumulative return and ending portfolio value
- CAGR, with the annualisation convention stated
- Annualised volatility
- Sharpe ratio, with the risk-free-rate assumption stated
- Maximum drawdown
- Number of trades and turnover
- Win rate
- Profit factor
- Average win and average loss per completed trade
- Total transaction costs

State the cash-return assumption. If cash earns zero, say so explicitly. Define how annualisation handles irregular timestamps; do not automatically treat every observation as exactly one hour.

### Benchmarks

Compare against:

1. **Buy and hold:** invest in USO over the same test interval, using comparable execution assumptions and costs.
2. **Cash:** remain uninvested throughout the test period.
3. **Previous-direction baseline:** use the previous observed price movement to predict the next direction.

Use the same test dates and comparable cost assumptions for all benchmarks.

## Robustness and sensitivity analysis

A single chronological holdout is not sufficient to establish robustness. Perform the following checks using training/validation data for model and strategy selection, reserving the final test period for the final evaluation.

### 1. Walk-forward evaluation

Use expanding-window or rolling-window validation:

- Train on an initial historical period.
- Generate predictions for the next unseen period.
- Move the evaluation window forward.
- Refit using only data available before each prediction period.
- Combine the out-of-sample returns from all evaluation windows.

Keep a purge gap consistent with the target horizon at each boundary. Do not use future observations to calculate features, fit preprocessing steps or select model parameters.

### 2. Transaction-cost sensitivity

Repeat the backtest under several plausible cost assumptions:

- Zero costs (diagnostic upper bound, not a realistic result).
- Low-cost scenario.
- Base-case scenario.
- High-cost scenario.

Report how net cumulative return, Sharpe ratio and drawdown change. Justify cost levels using available market information or label them clearly as assumptions.

### 3. Signal-threshold sensitivity

If the model exposes class probabilities, test different probability thresholds for entering a long position instead of relying only on the default 0.5 cutoff.

Compare a small set of thresholds around 0.5. Select thresholds using training/validation data only; do not choose the best threshold based on final test performance.

### 4. Feature ablation

Retrain the model after removing one feature at a time, then compare it with the full feature set. Also consider groups of related features:

- Return features: `internal_return`, `cc_return`.
- Volume features: `relative_volume`, `hourly_relative_volume`.
- Time features: `hour`, `time_gap`.
- Volatility: `volatility`.

Compare predictive metrics and net trading metrics. Feature importance from a tree or forest is model-specific and does not prove independent predictive value.

### 5. Model and parameter sensitivity

Compare the selected Decision Tree and Random Forest with simpler alternatives and nearby hyperparameter settings. Record whether conclusions depend on one particular configuration.

Avoid repeatedly tuning against the final test set. If test results are used to change the strategy, that period is no longer an untouched final evaluation.

### 6. Market-regime analysis

Where sample size permits, compare results across different market conditions, for example:

- High- and low-volatility periods.
- Rising and falling USO price periods.
- Periods with unusually large time gaps or trading interruptions.

Define regime labels using information available at the time. Report the number of observations and trades in each regime so small samples are not overinterpreted.

### 7. Statistical uncertainty

Where practical, estimate uncertainty around strategy performance using a block bootstrap or another method that preserves some time dependence. Ordinary independent resampling of hourly returns can understate uncertainty because financial returns are not necessarily independent.

## Reproducibility

The project uses Python and the following libraries:

- pandas
- NumPy
- Matplotlib
- scikit-learn
- joblib

The fitted grid searches and selected models are saved with `joblib`. For reproducibility, record the dataset version, date range, feature definitions, target definition, split dates, model parameters, random seeds, execution assumptions and cost assumptions alongside each experiment.

## Current conclusions

The tested classifiers achieve modest improvements over the majority-class baseline on the held-out period. The best test balanced accuracy among the reported models is approximately 0.524, achieved by the Decision Tree tuned with `max_leaf_nodes=7`.

These results suggest limited directional predictive performance so far. They do not establish that the model can generate positive net returns. That question remains open until the execution-aligned target, out-of-sample backtest, transaction costs, benchmarks and robustness checks have been implemented and evaluated.
