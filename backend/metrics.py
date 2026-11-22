"""Classification metrics, results assembly and report figures (no TensorFlow needed)."""
import json
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def evaluate_predictions(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    rows = cm.sum(1, keepdims=True).astype(float)
    norm = np.divide(cm.astype(float), rows, out=np.zeros(cm.shape), where=rows > 0)
    return {"accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
            "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
            "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(fw),
            "per_class": [{"class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate(classes)],
            "confusion_matrix": cm.tolist(), "confusion_matrix_norm": norm.round(4).tolist()}


def assemble_results(dataset_info, model_results, classes, config):
    best = max(model_results, key=lambda m: m["f1_macro"])["name"]
    return {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "classes": classes,
            "dataset": dataset_info, "config": config, "best_model": best, "models": model_results}


def save_results(results, *paths):
    for p in paths:
        p.write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def save_figures(results, outdir):
    classes, models = results["classes"], results["models"]
    for m in models:
        fig, ax = plt.subplots(figsize=(6, 5))
        cm, nm = np.array(m["confusion_matrix"]), np.array(m["confusion_matrix_norm"])
        ax.imshow(nm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]:,}\n{nm[i, j]:.1%}", ha="center", va="center",
                        color="white" if nm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(len(classes)), classes, rotation=25); ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix: {m['name']}")
        fig.tight_layout(); fig.savefig(outdir / f"confusion_{m['key']}.png", dpi=150); plt.close(fig)
    names = [m["name"] for m in models]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    w = 0.2
    for k, (lab, key) in enumerate([("Accuracy", "accuracy"), ("Precision (macro)", "precision_macro"),
                                    ("Recall (macro)", "recall_macro"), ("F1 (macro)", "f1_macro")]):
        ax[0].bar(np.arange(len(models)) + k * w, [m[key] for m in models], w, label=lab)
    ax[0].set_xticks(np.arange(len(models)) + 1.5 * w, names); ax[0].legend(); ax[0].set_title("Test-set metrics")
    ax[1].bar(names, [m["training_time_s"] for m in models], color="tab:orange")
    ax[1].set_title("Training time (seconds)")
    fig.tight_layout(); fig.savefig(outdir / "model_comparison.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for m in models:
        ax[0].plot(m["history"]["val_loss"], label=m["name"]); ax[1].plot(m["history"]["val_accuracy"], label=m["name"])
    ax[0].set_title("Validation loss"); ax[1].set_title("Validation accuracy"); ax[0].legend()
    for a in ax: a.set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(outdir / "training_curves.png", dpi=150); plt.close(fig)
