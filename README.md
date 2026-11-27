# StreamingAnalysis

Code, data and results for a bachelor's thesis on **knowledge extraction from network traffic flows using data mining techniques**.

The project has three parts:

1. **Dataset:** flows are extracted from public packet traces of the MAWI archive.
2. **Analysis:** TCP, UDP and ICMP flows are analysed separately for each day, and the days are then compared.
3. **Prediction:** LSTM networks predict the total size in bytes of each TCP flow from the flows that came before it.

![Flows per protocol and day](Results/week_protocols.png)

## Dataset

The flows come from the [MAWI Working Group Traffic Archive](https://mawi.wide.ad.jp/mawi/) of the WIDE project, **samplepoint-F**. This sampling point monitors the transit link of WIDE in Tokyo and, since 2023, also the link to the BBIX exchange point. Each daily trace covers 15 minutes, from 14:00 to 14:15 Japan time. IP addresses are anonymised with Crypto-PAn, and the mapping is consistent only within one daily trace, so the same address on two different days is not necessarily the same host.

**Sample:**
- 7 consecutive days, from Tuesday 9 to Monday 15 June 2026 (five weekdays and one weekend).
- Only the first minute of each trace is used (14:00–14:01).
- From that minute, the first **5,000 flows** to start are kept, whatever their protocol, so the dataset has **35,000 flows** in total.

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

Packets and volume count the packets of the 5,000 flows within the first minute of each trace.

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
        ├── week_lstm.csv
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

The "first N flows" are the N flows that start first. All their packets are counted until the end of the file, or until `--until`, so every flow kept is complete within the window. Packet size is the original frame length (`frame.len`).

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

Each protocol group (TCP, UDP, ICMP) is analysed separately. The target variable is `Bytes`, the total size of a flow.

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

**How to run it:** open it in Google Colab with the button above and choose *Runtime → Run all*. The notebook clones this repository by itself to get the CSV files. At the end it downloads all its outputs as one file, `week_lstm_results.zip`. If Colab shows an older version of the notebook, close the tab and open the link again.

To run it locally, install the requirements below, start `jupyter notebook` in the repository folder and run all cells.

**What it does:**
- **Data:** reads the seven CSV files and keeps only the TCP flows, sorted by start time (as in `analysis.py`).
- **Split:** each day is handled separately. The first 80% of its flows are used for training and the last 20% for testing, so the test flows are the same as those of the baselines in `Results/TCP/week_baselines.csv`.
- **Features:** chosen by Spearman correlation on the training part only. A field is kept only if its correlation with every field already kept is at most 0.9. The result is `Bytes`, `Bytes B → A` and `Duration`. The `Bits/s` fields are not used, because they are empty for flows shorter than 5 ms.
- **Preparation:** `log(1 + x)` and scaling to [0, 1] with `MinMaxScaler`, fitted on the training part only. Each sample is a window of the 100 previous flows, and the target is the `Bytes` of the next flow.
- **Models:** the three DL4NTP architectures with 50 LSTM units, 5 epochs and batch size 100. Simple LSTM is trained with MSE. Bidirectional and Stacked LSTM use the DL4NTP `rmse` loss, which with a single output equals the mean absolute error.
- **Evaluation:** MAE, RMSE and R² (the square of the correlation between predicted and actual values, as in DL4NTP), together with three baselines: the previous flow, the median of the previous 100 flows and the median of the training flows.
- **Reproducibility:** the random seed is fixed (42), so every run gives the same results.

All settings (days, features, correlation limit, window, epochs, batch size) are in the *Settings* cell. To run a single day, set for example `DAYS = ["20260609"]`.

**Outputs** (`DL4NTP/Results Prediction/`). Every figure has a caption underneath that explains what it shows.

| File | Content |
|---|---|
| `week_lstm.csv` | MAE, RMSE and R² of every model and baseline for each day |
| `week_lstm_split.png` | all TCP flows of each day, with the training part and the test part |
| `week_lstm_predictions.png` | actual and predicted bytes of the test flows, the three models together |
| `week_lstm_predictions_simple.png`, `_bidirectional.png`, `_stacked.png` | the same for one model at a time, with the daily MAE in the title of each panel |
| `week_lstm_loss.png` | training loss per epoch, for each model and day |

**Results** (mean over the 7 days, test flows only):

| Model | MAE (bytes) | RMSE (bytes) | R² |
|---|---|---|---|
| Previous flow | 747,781 | 5,746,641 | 0.000 |
| Median of previous 100 flows | 377,243 | 4,066,284 | 0.003 |
| Training median | 377,249 | 4,066,278 | – |
| Simple LSTM | 377,263 | 4,066,270 | 0.001 |
| Bidirectional LSTM | 377,243 | 4,066,285 | 0.003 |
| Stacked LSTM | 377,243 | 4,066,284 | 0.003 |

The three LSTMs reach the level of the median baselines and halve the error of the previous-flow baseline. They predict the typical flow size but not the few very large flows, and R² stays below 0.01. This agrees with the analysis, where the correlation of a flow with the next one is close to zero.

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
