"""Settings for the Weather Anomaly deep-learning project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "weather-anomalies-1964-2013.csv"
ART = ROOT / "backend" / "artifacts"
PUBLIC = ROOT / "frontend" / "public"
FIGS = ROOT / "report_figures"
for _p in (ART, PUBLIC, FIGS):
    _p.mkdir(parents=True, exist_ok=True)

CLASSES = ["Strong Cold", "Weak Cold", "Weak Hot", "Strong Hot"]
TRAIN_END_YEAR, VAL_END_YEAR = 2005, 2009          # train 1964-2005, validation 2006-2009, test 2010-2013
NUM_FEATURES = ["max_temp", "min_temp", "temp_range", "temp_mean", "latitude", "longitude", "doy_sin", "doy_cos"]
EMB_DIM = 8
SEED = 42

MODEL_SPECS = {
    "ann": ("ANN", "Feed-forward network: 128-64-32 ReLU layers with dropout"),
    "dnn": ("Deep Residual ANN", "Three residual blocks (128 units) with batch normalisation"),
    "cnn": ("CNN-1D", "1-D convolutions over the feature vector, then global max pooling"),
    "gru": ("GRU", "GRU (48 units) reading the feature vector as a short sequence"),
}
