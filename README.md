# StreamingAnalysis

Code, data and results for a bachelor's thesis on **knowledge extraction from network traffic flows using data mining techniques**.

The project has three parts:

1. **Dataset:** flows are extracted from public packet traces of the MAWI archive.
2. **Analysis:** TCP, UDP and ICMP flows are analysed separately for each day, and the days are then compared.
3. **Prediction:** LSTM networks try to predict the bytes of each TCP flow from the flows that came before it.

![Flows per protocol and day](Results/week_protocols.png)

## Dataset

The flows come from the [MAWI Working Group Traffic Archive](https://mawi.wide.ad.jp/mawi/) of the WIDE project, **samplepoint-F**. This sampling point monitors the transit link of WIDE in Tokyo and, since 2023, also the link to the BBIX exchange point. Each daily trace covers 15 minutes, from 14:00 to 14:15 Japan time. IP addresses are anonymised with Crypto-PAn, and the mapping is consistent only within one daily trace, so the same address on two different days is not necessarily the same host.

**Sample:**
- 7 consecutive days, from Tuesday 9 to Monday 15 June 2026 (five weekdays and one weekend).
- Only the first minute of each trace is used (14:00–14:01).
- From that minute, the first **5,000 flows** to start are kept, whatever their protocol, so the dataset has **35,000 flows** in total. All of them start within the first 29–45 ms of the trace.

| Date | Day | Flow starts (ms) | Packets | Volume (GB) | IPv4 / IPv6 (%) |
|---|---|---|---|---|---|
| 09/06/2026 | Tuesday | 0–29.6 | 8,625,150 | 13.29 | 92.2 / 7.7 |
| 10/06/2026 | Wednesday | 0–33.3 | 9,739,992 | 14.25 | 91.8 / 8.0 |
| 11/06/2026 | Thursday | 0–33.4 | 6,657,689 | 7.87 | 90.8 / 8.9 |
| 12/06/2026 | Friday | 0–29.3 | 7,725,432 | 9.45 | 93.3 / 6.5 |
| 13/06/2026 | Saturday | 0–42.4 | 4,055,512 | 4.30 | 95.7 / 4.1 |
| 14/06/2026 | Sunday | 0–45.2 | 4,041,152 | 4.85 | 96.1 / 3.6 |
| 15/06/2026 | Monday | 0–33.5 | 6,713,875 | 8.66 | 92.3 / 7.4 |
| **Total** | | | **47,558,802** | **62.67** | |

Packets and volume count the packets of the 5,000 flows within the first minute of each trace. The rest of the IPv4 / IPv6 percentage (0.1–0.3%) is non-IP traffic, such as ARP.

**What `Bytes` measures:** the bytes of a flow **within the first minute** of the trace, not its total size. Flows that continue after 14:01 are cut at 60 s, and flows that were already in progress at 14:00 are counted only from 14:00. Between 6% and 19% of the TCP flows of each day are still active at the end of the minute.

## Repository structure

```
StreamingAnalysis/
├── pcap_first_flows.py        # MAWI pcap -> CSV with the first N flows (all protocols)
├── analysis.py                # statistical analysis per protocol and day, comparison of days
├── Conversations/             # the dataset: one CSV per day
│   ├── first_5000_flows_20260609.csv
│   ├── first_5000_flows_20260609.png   # flows, packets and bytes per protocol for that day
│   └── ...                             # same for 10/06 - 15/06
├── Results/                   # output of analysis.py
│   ├── week_protocols.csv / .png       # protocol mix per day
│   ├── TCP/
│   │   ├── 20260609/ ... 20260615/     # analysis of each day
│   │   └── week_*                      # comparison of the 7 days
│   ├── UDP/                            # same structure
│   └── ICMP/                           # same structure (ICMP and ICMPv6)
└── DL4NTP/
    ├── DL4NTP.ipynb           # LSTM prediction on the TCP flows (Simple, Bidirectional, Stacked)
    └── Results Prediction/    # output of the notebook
        ├── week_lstm.csv              # results per day
        ├── week_lstm_summary.csv      # mean of the 7 days per model
        ├── week_lstm_split.png
        ├── week_lstm_predictions.png
        ├── week_lstm_predictions_simple.png
        ├── week_lstm_predictions_bidirectional.png
        ├── week_lstm_predictions_stacked.png
        └── week_lstm_loss.png
```

## 1. Flow extraction: `pcap_first_flows.py`

The script reads a MAWI trace with `tshark` and keeps the first N flows (default 5,000), whatever their protocol. It writes them to a CSV file and draws a PNG with the protocol mix.

```bash
python3 pcap_first_flows.py 202606091400.pcap.gz first_5000_flows_20260609.csv --until 60
```

- `--n`: number of flows to keep (default 5000)
- `--until`: stop reading after this many seconds (60 = first minute)
- `--png`: name of the PNG file (default: same name as the CSV)

**How a flow is defined:**
- **TCP:** one flow per connection (`tcp.stream`), as in Wireshark's *Statistics → Conversations → TCP*.
- **UDP:** one flow per `udp.stream`, i.e. per pair of addresses and ports.
- **Protocols without ports (ICMP, ICMPv6, GRE, ESP, tunnels):** one flow per pair of IP addresses and protocol.
- **Non-IP traffic (e.g. ARP):** one flow per pair of MAC addresses and protocol.

The outer header decides the protocol. An ICMP error that carries a TCP header counts as ICMP, and a tunnel counts as one flow.

The "first N flows" are the N flows whose first packet appears first in the trace. All their packets are counted until the end of the file, or until `--until`, so every flow kept is complete within the window; a flow that continues after the window is cut at its end. Packet size is the original frame length (`frame.len`).

**Columns:** `Network, Protocol, Address A, Port A, Address B, Port B, Packets, Bytes, Stream ID, Packets A → B, Bytes A → B, Packets B → A, Bytes B → A, Rel Start, Duration, Bits/s A → B, Bits/s B → A`

- A is the source of the flow's first packet.
- `Rel Start` is measured in seconds from the start of the trace.
- `Bits/s` is left empty when `Duration` < 5 ms, as in Wireshark.
- Rows are sorted by `Rel Start`.

## 2. Analysis: `analysis.py`

Run it on all seven days:

```bash
python3 analysis.py Conversations/first_5000_flows_*.csv
```

Or run it on a single day:

```bash
python3 analysis.py Conversations/first_5000_flows_20260609.csv
```

Each protocol group (TCP, UDP, ICMP) is analysed separately. The target variable is `Bytes`, the bytes of a flow within the first minute.

**Per day** (`Results/<protocol>/<date>/<date>_*`):

| File | Content |
|---|---|
| `describe.csv`, `statistics.csv` | descriptive statistics of the flow features |
| `DistributionOfDatasetColumns.png` | distributions of the flow features |
| `correlation.csv`, `correlation_analysis.png` | Spearman correlation between the features of the same flow (redundant inputs) |
| `lag_correlation.csv`, `lag_correlation.png` | Spearman correlation of each feature of flow *t* with the bytes of flow *t+k* (k = 0, 1, 10, 100) |
| `autocorrelation.csv`, `autocorrelation.png` | autocorrelation of the flow sizes |
| `TimeSeries.png`, `per_0.1s.csv` | traffic aggregated per 0.1 s |
| `baselines.csv` | MAE and RMSE of simple baselines on the last 20% of the flows (the models the LSTM must beat) |
| `summary.csv` | summary of the day |

**Note on the time-based outputs:** all the flows of a day start within 29–45 ms, so the aggregation per 0.1 s has a single interval. As a result `per_0.1s.csv` has one row, the per-0.1 s values in `autocorrelation.csv` and `summary.csv` are empty, and the lower two panels of `TimeSeries.png` are empty (only the upper panel, the bytes of each flow, is meaningful). In the same way, `New flows per second` in `summary.csv` is the rate at which flows are first seen in those few milliseconds, not the rate at which new connections arrive. These outputs are kept for completeness but are not used in the thesis.

**Comparison of the 7 days** (`Results/<protocol>/week_*`):

| File | Content |
|---|---|
| `week_summary.csv` | key figures per day, with weekday and weekend means |
| `week_by_day.png`, `week_distributions.png` | the days side by side |
| `week_similarity_ks.csv`, `week_similarity.png` | Kolmogorov–Smirnov distance between the byte distributions of each pair of days |
| `week_weekday_vs_weekend.csv` | weekdays compared with the weekend |
| `week_heavy_hitters.csv`, `week_heavy_hitters.png`, `week_top_pairs.csv` | the largest flows and the address pairs behind them |
| `week_baselines.csv` | baselines under different train/test splits by day |

The protocol mix of all days is written to `Results/week_protocols.csv` and `Results/week_protocols.png`.

## 3. Prediction: `DL4NTP/DL4NTP.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasnastos/StreamingAnalysis/blob/main/DL4NTP/DL4NTP.ipynb)

The notebook is based on the DL4NTP code of Fleischer & Myerson (link below), adapted to the MAWI TCP flows in `Conversations/`. It compares three architectures: **Simple LSTM**, **Bidirectional LSTM** and **Stacked LSTM**.

**How to run it:** open it in Google Colab with the button above and choose *Runtime → Run all*. The notebook clones this repository by itself to get the CSV files. It writes its outputs to `DL4NTP/Results Prediction/` and at the end downloads them all as one file, `week_lstm_results.zip`. If Colab shows an older version of the notebook, close the tab and open the link again.

To run it locally, install the requirements below, start `jupyter notebook` in the repository folder and run all cells.

**What it does:**
- **Data:** reads the seven CSV files and keeps only the TCP flows, sorted by start time (as in `analysis.py`).
- **Split:** each day is handled separately. The first 80% of its flows are used for training and the last 20% for testing, so the test flows are the same as those of the baselines in `Results/TCP/week_baselines.csv`.
- **Features:** chosen by Spearman correlation on the training part only. A field is kept only if its correlation with every field already kept is at most 0.9. The result is `Bytes`, `Bytes B → A` and `Duration`. The `Bits/s` fields are not used, because they are empty for flows shorter than 5 ms.
- **Preparation:** `log(1 + x)` and scaling to [0, 1] with `MinMaxScaler`, fitted on the training part only. Each sample is a window of the 100 previous flows, and the target is the `Bytes` of the next flow.
- **Models:** the three DL4NTP architectures with 50 LSTM units, 5 epochs and batch size 100. Simple LSTM is trained with MSE. Bidirectional and Stacked LSTM use the DL4NTP `rmse` loss, which with a single output equals the mean absolute error.
- **Evaluation:** on the test flows of each day:
  - **MAE** and **RMSE**, in bytes;
  - **R²**, the coefficient of determination 1 − SSE/SST. A value of 0 means no better than the mean of the test flows, and a negative value means worse. (DL4NTP uses the square of the correlation instead, which is always ≥ 0 and does not penalise an almost constant prediction.)
  - **MAE log10**, the mean of |log10(1 + actual) − log10(1 + predicted)|, i.e. the error in orders of magnitude (1 = ten times off). On this scale the few very large flows do not dominate.

  Four baselines are evaluated on the same flows: the previous flow, the median of the previous 100 flows, the median of the training flows, and always predicting 0 bytes.
- **Offline experiment:** the inputs (bytes and duration of the previous flows) are known only after those flows end, so this is not a real-time prediction.
- **Reproducibility:** the random seed is fixed (42), so every run gives the same results.

All settings (days, features, correlation limit, window, epochs, batch size) are in the *Settings* cell. To run a single day, set for example `DAYS = ["20260609"]`.

**Outputs** (`DL4NTP/Results Prediction/`). Every figure has a caption underneath that explains what it shows.

| File | Content |
|---|---|
| `week_lstm.csv` | MAE, RMSE, R² and MAE log10 of every model and baseline for each day (7 models × 7 days) |
| `week_lstm_summary.csv` | the same metrics averaged over the 7 days, one row per model (the table below) |
| `week_lstm_split.png` | all TCP flows of each day, with the training part and the test part |
| `week_lstm_predictions.png` | actual and predicted bytes of the test flows, the three models together |
| `week_lstm_predictions_simple.png`, `_bidirectional.png`, `_stacked.png` | the same for one model at a time; the title of each panel gives the daily MAE of the model, of the training median and of always predicting 0 bytes |
| `week_lstm_loss.png` | training loss per epoch, for each model and day |

**Results.** The prediction is made separately for each day: every model is trained on the first 80% of a day's TCP flows and tested on the last 20% of the same day, which gives 7 days × 3 models = 21 trainings. The daily results are in `week_lstm.csv`. The table below (`week_lstm_summary.csv`) is the mean of the 7 daily results for each model, not a separate prediction over the whole week.

| Model | MAE (bytes) | RMSE (bytes) | R² | MAE log10 |
|---|---|---|---|---|
| Previous flow | 747,781 | 5,746,641 | -1.022 | 1.152 |
| Median of previous 100 flows | 377,243 | 4,066,284 | -0.013 | 0.816 |
| Training median | 377,249 | 4,066,278 | -0.013 | 0.826 |
| Always 0 bytes | 377,300 | 4,066,296 | -0.013 | 2.588 |
| Simple LSTM | 377,263 | 4,066,270 | -0.013 | 0.870 |
| Bidirectional LSTM | 377,243 | 4,066,285 | -0.013 | 0.822 |
| Stacked LSTM | 377,243 | 4,066,284 | -0.013 | 0.824 |

**What the results show:** the LSTMs do not learn more than a typical flow size.

- Their predictions stay between 57 and 1,047 bytes, while the test flows range from 54 bytes to 245 MB.
- In bytes, the MAE of every LSTM is only 4–60 bytes per day lower than that of always predicting 0 bytes. The MAE is set by the few very large flows: the top 1% of the test flows carries 44–99% of their bytes, depending on the day. For the same reason the previous-flow baseline has twice the MAE: it sometimes repeats a very large flow.
- R² is slightly negative for every model, so no model does better than the mean of the test flows.
- On the log10 scale, where the large flows do not dominate, Bidirectional and Stacked LSTM are level with the training median and Simple LSTM is worse. No LSTM beats the median of the previous 100 flows.

This agrees with the analysis, where the Spearman correlation between the bytes of a flow and those of the next one is only 0.03–0.07: the bytes of a flow cannot be predicted from the flows that started before it.

![Actual and predicted bytes of the test flows](DL4NTP/Results%20Prediction/week_lstm_predictions.png)

## Requirements

- Python 3 with `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`
- `tshark` (part of Wireshark) for `pcap_first_flows.py`
- `tensorflow` / `keras` for the LSTM notebook (already installed in Google Colab)

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn tensorflow
```

## Data source

- MAWI Working Group Traffic Archive: https://mawi.wide.ad.jp/mawi/
- samplepoint-F traces of 2026: https://mawi.wide.ad.jp/mawi/samplepoint-F/2026/
- K. Cho, K. Mitsuya, A. Kato, *Traffic Data Repository at the WIDE Project*, USENIX Annual Technical Conference, FREENIX Track, 2000.

## Links
* [https://github.com/nb0309/Network-Traffic-Analysis-using-Machine-learning/blob/main/anomalydetection.ipynb](https://github.com/nb0309/Network-Traffic-Analysis-using-Machine-learning/blob/main/anomalydetection.ipynb)
* [https://github.com/shikhar-srivastava/Network-Traffic-Classification](https://github.com/shikhar-srivastava/Network-Traffic-Classification)
* [https://github.com/biagiom/spark-network-traffic-classifier](https://github.com/biagiom/spark-network-traffic-classifier)
* [https://github.com/fle1scha/Deep-Learning-Network-Traffic-Prediction](https://github.com/fle1scha/Deep-Learning-Network-Traffic-Prediction)
