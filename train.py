"""Train and compare the deep-learning models, then export results for the React dashboard.

    python train.py --epochs 3 --max-train 100000      # quick test
    python train.py                                    # full run (default: 600k training rows)
    python train.py --models ann cnn                   # subset of models
"""
import argparse
import random
import time

import numpy as np

from backend.config import ART, CLASSES, DATA_CSV, FIGS, MODEL_SPECS, PUBLIC, SEED
from backend.metrics import assemble_results, evaluate_predictions, save_figures, save_results
from backend.prepare_data import build_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA_CSV))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-train", type=int, default=600_000)
    ap.add_argument("--max-val", type=int, default=100_000)
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    a = ap.parse_args()

    import tensorflow as tf
    from backend.models import build_model
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    print("Loading and cleaning data ...")
    data, info, n_st = build_datasets(a.csv, a.max_train, a.max_val)
    tr, va, te = data["train"], data["val"], data["test"]
    print(f"rows: raw {info['rows_raw']:,} -> clean {info['rows_clean']:,}; "
          f"train {len(tr['y']):,}  val {len(va['y']):,}  test {len(te['y']):,}")

    counts = np.bincount(tr["y"], minlength=len(CLASSES))
    class_weight = {i: float((len(tr["y"]) / (len(CLASSES) * max(c, 1))) ** 0.5) for i, c in enumerate(counts)}
    xtr, xva, xte = ({"num": d["X"], "station": d["s"]} for d in (tr, va, te))

    results = []
    for key in a.models:
        tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
        name, desc = MODEL_SPECS[key]
        model = build_model(key, tr["X"].shape[1], n_st)
        cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5)]
        print(f"\n=== Training {name} ({model.count_params():,} parameters) ===")
        t0 = time.perf_counter()
        h = model.fit(xtr, tr["y"], validation_data=(xva, va["y"]), epochs=a.epochs, batch_size=a.batch,
                      class_weight=class_weight, callbacks=cbs, verbose=2)
        train_s = time.perf_counter() - t0                                  # training time (wall clock)
        t1 = time.perf_counter()
        prob = model.predict(xte, batch_size=4096, verbose=0)
        infer_ms = (time.perf_counter() - t1) / len(te["y"]) * 1e6          # ms per 1000 predictions
        m = evaluate_predictions(te["y"], prob.argmax(1), CLASSES)
        hist = {k: [float(v) for v in vals] for k, vals in h.history.items() if k in ("loss", "val_loss", "accuracy", "val_accuracy")}
        results.append({"key": key, "name": name, "description": desc, "params": int(model.count_params()),
                        "training_time_s": round(train_s, 1), "epochs_run": len(hist["loss"]),
                        "best_epoch": int(np.argmin(hist["val_loss"]) + 1), "inference_ms_per_1000": round(infer_ms, 2),
                        **m, "history": hist})
        model.save(ART / f"{key}.keras")
        print(f"{name}: accuracy {m['accuracy']:.4f}  macro-F1 {m['f1_macro']:.4f}  time {train_s:.0f}s")

    cfg = {"epochs_max": a.epochs, "batch_size": a.batch, "optimizer": "Adam (1e-3)", "loss": "sparse categorical cross-entropy",
           "class_weights": "sqrt-balanced", "early_stopping_patience": 3, "seed": SEED}
    final = assemble_results(info, results, CLASSES, cfg)
    save_results(final, ART / "results.json", PUBLIC / "results.json")
    save_figures(final, FIGS)
    print(f"\nBest model by macro-F1: {final['best_model']}")
    print("Results: frontend/public/results.json   Figures: report_figures/")


if __name__ == "__main__":
    main()
