MODULE D - CONNECTIONIST MODELS

Hopfield Network:
Stores and recalls 8x8 binary patterns after corruption.

RNN Anomaly Detection:
Uses the real SKAB industrial time-series dataset. The dataset must be downloaded manually and placed at data/skab/valve1_1.csv, or supplied with --dataset. The RNN uses 10-reading windows, standardized sensor features, chronological train/validation/test splits, and reports accuracy, precision, recall and F1.

Dataset:
https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab

Run:
python -m module_d_connectionist.rnn_anomaly --dataset data/skab/valve1_1.csv
