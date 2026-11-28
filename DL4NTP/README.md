# Results of the LSTM prediction

This folder holds the output of [`DL4NTP.ipynb`](../DL4NTP.ipynb). This page explains how to read `week_lstm.csv` and what it shows. The mean of the 7 days for each model is in `week_lstm_summary.csv` and in the main [README](../../README.md).

## How to read `week_lstm.csv`

Each row is one model on one day: 7 days × 7 models = 49 rows.

| Column | Meaning |
|---|---|
| `Day` | the day of the MAWI trace |
| `Model` | the model (see below) |
| `Test flows` | the number of test flows of that day, i.e. the last 20% of its TCP flows (473–529) |
| `MAE` | mean absolute error in bytes: on average, how many bytes the prediction is off for each test flow |
| `RMSE` | root mean squared error in bytes: like MAE, but each error is squared first, so large errors count much more |
| `R2` | coefficient of determination: 1 is a perfect prediction, 0 is no better than the mean of the test flows, below 0 is worse than that |
| `MAE log10` | mean error in orders of magnitude: 0.3 is about two times off, 1 is ten times off, 2 is a hundred times off; every flow counts the same, small or large |

The numbers are written with full precision and a dot as the decimal separator. For example, `852518.911153119` is 852,519 bytes, about 0.85 MB.

**The models:**
- **Simple LSTM**, **Bidirectional LSTM**, **Stacked LSTM**: the three neural networks, each trained on the first 80% of the day's TCP flows.
- **Previous flow**: predicts that each flow is as large as the flow before it.
- **Median of previous 100 flows**: predicts the median size of the 100 flows before it.
- **Train median**: always predicts the median size of the training flows.
- **Always 0 bytes**: always predicts 0. It learns nothing and is there only as a check on the metrics.

## What the table shows

1. **MAE and RMSE in bytes hardly separate the models.** On each day, all models except Previous flow are within 56–60 bytes of each other in MAE. Their MAE is about the mean size of the test flows of that day (for Always 0 bytes it is exactly that), which is why it changes so much from day to day: from 108,417 bytes on Saturday to 852,572 on Tuesday. It depends on how many of the few very large flows fall in the test part. RMSE is 6–19 times larger than MAE for the same reason.

2. **Previous flow is about twice as bad.** Its MAE is 1.96–2.00 times the best MAE of the day and its R² is about −1 (−1.006 to −1.038) on every day. Since a flow is almost unrelated to the one before it, copying the previous flow is even worse than always predicting 0 bytes.

3. **R² is slightly below 0 for all other models** (−0.003 to −0.032), so none of them does better than the mean of the test flows.

4. **MAE log10 is where the models differ.**
   - Always 0 bytes: 2.10–2.92, i.e. about 125 to 840 times off.
   - All the other models except Previous flow: 0.34–1.19.
   - The best model of each day is an LSTM on three days (Stacked on Tuesday and Thursday, Bidirectional on Monday) and one of the median baselines on the other four. Bidirectional and Stacked LSTM are always within 0.025 of the median of the previous 100 flows.
   - Simple LSTM is the weakest LSTM, and worse than the training median, on every day.
   - The weekend is easier (0.34–0.56) than the weekdays (0.73–1.19), because on Saturday and Sunday most TCP flows are small and alike.

**In short:** the LSTMs learn the typical size of a flow and do as well as a simple median of the previous flows, but no better. They cannot predict which flows will be very large, because the size of a flow is almost unrelated to the flows that came before it.
