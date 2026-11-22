# Weather Anomaly Classifier: Deep Learning + React Dashboard

Classifies each station-day in the **weather-anomalies-1964-2013** dataset (3.2 million US records) into
**Strong Cold, Weak Cold, Weak Hot, Strong Hot** with four deep-learning models, and compares them on:

**Accuracy · Precision · Recall · F1-score · Training time · Confusion matrix**

| Model | Idea |
|---|---|
| ANN | 128-64-32 dense network with dropout |
| Deep Residual ANN | three residual blocks with batch normalisation |
| CNN-1D | 1-D convolutions over the feature vector |
| GRU | recurrent layer reading the feature vector as a short sequence |

All four use the same inputs: max/min temperature, range, mean, latitude, longitude, day-of-year (sin/cos) and a learned **station embedding**.

## Important design decisions (mention these in your report and viva)

1. **`degrees_from_mean` is not used as an input.** The class label is derived from it; a quick tree-model check on a sample reached about 99.6% accuracy with it and about 90% without. Using it would be data leakage and make the comparison meaningless.
2. **Cleaning:** 9,153 rows removed (minimum temperature above maximum, or impossible temperatures).
3. **Chronological split:** train 1964-2005, validation 2006-2009, test 2010-2013. No random shuffling across time.
4. **Imbalance:** "Weak Hot" and "Weak Cold" dominate, so models use sqrt-balanced class weights and the dashboard reports **macro** (each class equal) and **weighted** averages. The best model is chosen by macro-F1. Expect the rare *Strong* classes to be the hardest; the confusion matrix shows this.
5. **Training time** = wall-clock time of `model.fit` on your computer. Report your hardware (CPU/GPU) next to it.

## Setup (Windows PowerShell)

Your laptop blocks `pip.exe`, so always call pip through Python:

```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
TensorFlow needs Python 3.10 or 3.11 (check with `python --version`). If TensorFlow will not install, train on Google Colab, then download `frontend/public/results.json` and the `backend/artifacts` folder.

## Run

1. Put the dataset at `data/weather-anomalies-1964-2013.csv`.
2. Train (quick test first, then the full run):
   ```
   python train.py --epochs 3 --max-train 100000
   python train.py
   ```
   Writes `frontend/public/results.json`, `backend/artifacts/*.keras` and report figures in `report_figures/`.
   Default uses 600k training rows for speed; use `--max-train 3000000` to use all training rows (slower).
3. Open the dashboard, **no npm needed**: double-click `standalone/dashboard.html`, click **Load results.json** and choose `frontend/public/results.json`.
4. Optional, full React development setup (needs Node.js 18+):
   ```
   cd frontend
   npm install
   npm run dev
   ```
   Open the address shown (usually http://localhost:5173); results load automatically.
5. Optional, live predictions in the **Predict** tab: in a second terminal run `python api.py`.

## Dashboard tabs

Overview (data, split, class balance, method notes) · Model comparison (table + charts for accuracy/precision/recall/F1 and training time) · Confusion matrix (per model, % or counts, per-class metrics) · Training curves · Predict.

## Folder structure

```
train.py  api.py  requirements.txt
backend/    config.py  prepare_data.py  models.py  metrics.py
frontend/   Vite + React app (src/App.jsx, tabs.jsx, charts.jsx)
standalone/ dashboard.html  (prebuilt, works without npm)
report_figures/  PNGs created by train.py (confusion matrices, comparison, curves)
```

## Report checklist

Use only numbers from your own run (`results.json`). Include: problem and dataset, cleaning and the leakage decision, split, architectures, the comparison table, confusion matrices, training curves, discussion of which model is best and why (and the rare-class weakness), limitations (US stations only, 4 classes derived from a fixed anomaly rule), and future work.
