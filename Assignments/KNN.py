import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support)

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "cifar-10" / "cifar-10-python" / "cifar-10-batches-py"
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent.parent
OUT_FILE = ROOT / (SCRIPT.stem + "_results.md")
IMG_DIR = ROOT / (SCRIPT.stem + "_results_plots")
IMG_DIR.mkdir(exist_ok=True)

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Could not find data directory: {DATA_DIR}")

# Load CIFAR-10
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_batch(name):
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing batch file: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"].astype(np.float32), np.array(d[b"labels"])

X_train, y_train = [], []
for i in range(1, 6):
    X, y = load_batch("data_batch_%d" % i)
    X_train.append(X)
    y_train.append(y)
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
X_test, y_test = load_batch("test_batch")

# use a subset
N_TRAIN, N_TEST, SEED = 5000, 1000, 0
np.random.seed(SEED)
idx = np.random.permutation(len(X_train))[:N_TRAIN]
X_train, y_train = X_train[idx], y_train[idx]
X_test, y_test = X_test[:N_TEST], y_test[:N_TEST]

# KNN classifier
class KNN:
    def __init__(self, k=1, metric="L1"):
        self.k = k
        self.metric = metric

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def _distances(self, x):
        if self.metric == "L1":
            return np.sum(np.abs(self.Xtr - x), axis=1)
        elif self.metric == "L2":
            return np.sqrt(np.sum((self.Xtr - x) ** 2, axis=1))
        raise ValueError("metric must be 'L1' or 'L2'")

    def predict_multi_k(self, X, ks):
        preds = {k: np.zeros(X.shape[0], dtype=self.ytr.dtype) for k in ks}
        for i in range(X.shape[0]):
            order = np.argsort(self._distances(X[i, :]))
            for k in ks:
                preds[k][i] = np.bincount(self.ytr[order[:k]]).argmax()
        return preds

    def predict(self, X):
        return self.predict_multi_k(X, [self.k])[self.k]

# 5-fold cross-validation: returns per-fold accuracies for every k
FOLDS = 5
def cross_validation(X, y, ks, metric, folds=FOLDS):
    X_folds = np.array_split(X, folds)
    y_folds = np.array_split(y, folds)
    fold_accs = {k: [] for k in ks}
    for f in range(folds):
        X_val, y_val = X_folds[f], y_folds[f]
        X_tr = np.concatenate(X_folds[:f] + X_folds[f + 1:])
        y_tr = np.concatenate(y_folds[:f] + y_folds[f + 1:])
        model = KNN(metric=metric)
        model.train(X_tr, y_tr)
        preds = model.predict_multi_k(X_val, ks)
        for k in ks:
            fold_accs[k].append(accuracy_score(y_val, preds[k]))
    return fold_accs

K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
METRICS = ["L1", "L2"]

cv_results = {}
test_results = {}

for metric in METRICS:
    print("Running %s ..." % metric)
    fold_accs = cross_validation(X_train, y_train, K_VALUES, metric)
    cv_results[metric] = fold_accs

    best_k = max(K_VALUES, key=lambda k: np.mean(fold_accs[k]))
    best_acc = np.mean(fold_accs[best_k])

    model = KNN(best_k, metric)
    model.train(X_train, y_train)
    pred = model.predict(X_test)

    scores = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, pred, average="macro", zero_division=0),
    }
    test_results[metric] = {"best_k": best_k, "best_cv": best_acc, "pred": pred, "scores": scores}


# Build results.md
md = []

def md_table(header, rows, bold_rows=()):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for i, r in enumerate(rows):
        cells = [str(c) for c in r]
        if i in bold_rows:
            cells = ["**%s**" % c if c != "" else c for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

# Add figure as a PNG in IMG_DIR and link it to the markdown file
def add_figure(fig, filename, alt):
    fig.savefig(IMG_DIR / filename, dpi=150, bbox_inches="tight")
    md.append("![%s](%s/%s)\n" % (alt, IMG_DIR.name, filename))
    plt.close(fig)

md.append("# KNN on CIFAR-10 - Results\n")
md.append("**Setup:** %d training images (random subset, seed %d), %d test images, "
          "%d-fold cross-validation, k in %s, distances: %s.\n"
          % (N_TRAIN, SEED, N_TEST, FOLDS, K_VALUES, ", ".join(METRICS)))
md.append("Std = standard deviation of the %d fold accuracies. "
          "Precision / Recall / F1 are macro averages over the 10 classes. "
          "The best k per distance is shown in bold.\n" % FOLDS)

# Table 1: cross-validation per metric
md.append("## 1. Cross-validation (choosing k)\n")
for metric in METRICS:
    fa = cv_results[metric]
    bk = test_results[metric]["best_k"]
    header = ["k", "Mean acc.", "Std"] + ["Fold %d" % (i + 1) for i in range(FOLDS)]
    rows = [[k, "%.4f" % np.mean(fa[k]), "%.4f" % np.std(fa[k])] + ["%.4f" % a for a in fa[k]] for k in K_VALUES]
    md.append("### %s distance\n" % metric)
    md.append(md_table(header, rows, bold_rows=[K_VALUES.index(bk)]))

# Table 2: test results
header = ["Distance", "Best k", "CV acc.", "Test acc.", "Precision", "Recall", "F1-score"]
rows = []
for metric in METRICS:
    t = test_results[metric]
    s = t["scores"]
    rows.append([metric, t["best_k"], "%.4f" % t["best_cv"], "%.4f" % s["Accuracy"], "%.4f" % s["Precision"], "%.4f" % s["Recall"], "%.4f" % s["F1-score"]])
md.append("## 2. Test-set results\n")
md.append(md_table(header, rows))

# Table 3: per-class scores
md.append("## 3. Per-class scores on the test set\n")
for metric in METRICS:
    p, r, f, sup = precision_recall_fscore_support(y_test, test_results[metric]["pred"], labels=range(10), zero_division=0)
    header = ["Class", "Precision", "Recall", "F1-score", "Support"]
    rows = [[CLASS_NAMES[i], "%.3f" % p[i], "%.3f" % r[i], "%.3f" % f[i], int(sup[i])] for i in range(10)]
    md.append("### %s (k=%d)\n" % (metric, test_results[metric]["best_k"]))
    md.append(md_table(header, rows))

# PLOTS
md.append("## 4. Plots\n")

# 1a Cross-validation on k (dots = folds, line = mean, bars = std)
fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 4.5), sharey=True)
for ax, metric in zip(axes, METRICS):
    fa = cv_results[metric]
    for k in K_VALUES:
        ax.scatter([k] * len(fa[k]), fa[k], color="navy", s=18, zorder=3)
    means = np.array([np.mean(fa[k]) for k in K_VALUES])
    stds = np.array([np.std(fa[k]) for k in K_VALUES])
    ax.errorbar(K_VALUES, means, yerr=stds, color="royalblue", capsize=4, linewidth=1.5, zorder=2)
    bk = test_results[metric]["best_k"]
    ax.axvline(bk, color="red", linestyle="--", alpha=0.6, label="best k = %d" % bk)
    ax.set_title("Cross-validation on k (%s distance)" % metric)
    ax.set_xlabel("k")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Cross-validation accuracy")
plt.tight_layout()
md.append("### Cross-validation on k\n\nEach dot is one fold, the line goes through the "
          "mean, the bars show the standard deviation.\n")
add_figure(fig, "cv_on_k.png", "Cross-validation on k")

# 1b L1 vs L2
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = {"L1": "tab:blue", "L2": "tab:orange"}
for metric in METRICS:
    fa = cv_results[metric]
    means = np.array([np.mean(fa[k]) for k in K_VALUES])
    stds = np.array([np.std(fa[k]) for k in K_VALUES])
    ax.errorbar(K_VALUES, means, yerr=stds, marker="o", capsize=4, color=colors[metric], label=metric)
ax.set_xlabel("k")
ax.set_ylabel("Cross-validation accuracy")
ax.set_title("L1 vs L2: mean CV accuracy (+/- 1 std)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
md.append("### L1 vs L2\n")
add_figure(fig, "cv_l1_vs_l2.png", "L1 vs L2")

# 2 Confusion matrices (counts + row-normalised)
md.append("### Confusion matrices\n")
for metric in METRICS:
    pred = test_results[metric]["pred"]
    bk = test_results[metric]["best_k"]
    cm = confusion_matrix(y_test, pred, labels=range(10))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, data, title, fmt in [(axes[0], cm, "Counts", "d"), (axes[1], cm_norm, "Normalised by true class (recall)", ".2f")]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        thresh = data.max() / 2
        for i in range(10):
            for j in range(10):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8, color="white" if data[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Confusion matrix - KNN (k=%d, %s)" % (bk, metric))
    plt.tight_layout()
    md.append("**%s (k=%d)**\n" % (metric, bk))
    add_figure(fig, "confusion_matrix_%s.png" % metric, "Confusion matrix %s" % metric)

# 3 Per-class precision / recall / F1
fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.5), sharey=True)
x = np.arange(10)
w = 0.27
for ax, metric in zip(axes, METRICS):
    p, r, f, _ = precision_recall_fscore_support(y_test, test_results[metric]["pred"], labels=range(10), zero_division=0)
    ax.bar(x - w, p, w, label="Precision")
    ax.bar(x, r, w, label="Recall")
    ax.bar(x + w, f, w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_title("Per-class scores (k=%d, %s)" % (test_results[metric]["best_k"], metric))
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Score")
axes[0].legend()
plt.tight_layout()
md.append("### Per-class scores\n")
add_figure(fig, "per_class_scores.png", "Per-class scores")

# 4 Overall test metrics, L1 vs L2
fig, ax = plt.subplots(figsize=(7, 4.5))
names = list(test_results["L1"]["scores"].keys())
x = np.arange(len(names))
w = 0.35
for i, metric in enumerate(METRICS):
    vals = [test_results[metric]["scores"][n] for n in names]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label="%s (k=%d)" % (metric, test_results[metric]["best_k"]))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, "%.3f" % v, ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Score")
ax.set_title("Test-set metrics (macro average)")
ax.grid(axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()
md.append("### Test-set metrics, L1 vs L2\n")
add_figure(fig, "test_metrics_comparison.png", "Test-set metrics")

# Write the output markdown file
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("Saved:", OUT_FILE)