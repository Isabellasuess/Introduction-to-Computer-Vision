import pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "cifar-10" / "cifar-10-python" / "cifar-10-batches-py"
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent.parent
OUT_FILE = ROOT / (SCRIPT.stem + "_results.md")
IMG_DIR = ROOT / (SCRIPT.stem + "_results_plots")
IMG_DIR.mkdir(exist_ok=True)

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Could not find data directory: {DATA_DIR}")

# Load CIFAR-10 (test batch)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
C = len(CLASS_NAMES)
SEED = 42

with open(DATA_DIR / "test_batch", "rb") as fo:
    batch = pickle.load(fo, encoding="bytes")
X_raw = np.asarray(batch[b"data"], dtype=np.uint8)  # (10000, 3072), RGB
y_all = np.asarray(batch[b"labels"], dtype=np.int64)
X_all = X_raw.astype(np.float64) / 255.0 - 0.5


def to_image(row):
    # 3072-vector (RGB) -> 32x32x3
    return np.asarray(row).reshape(3, 32, 32).transpose(1, 2, 0)


# Parameters: small random W, zero bias, fixed seed (SEED = 42)
rng = np.random.default_rng(SEED)
W = 0.01 * rng.standard_normal((C, 3072))
b = np.zeros(C)


# Scores, hinge loss, softmax: Score f(x) = Wx + b (one row of scores per image)
def scores_fn(W, b, X):
    return X @ W.T + b

# Multiclass SVM hinge loss  L_i = sum_{j != y_i} max(0, s_j - s_yi + 1)
def hinge_loss(scores, y, delta=1.0):
    n = len(y)
    margins = np.maximum(0, scores - scores[np.arange(n), y][:, None] + delta)
    margins[np.arange(n), y] = 0
    return margins.sum(axis=1)

# Softmax: scores -> probabilities (exponentiate, then normalize)
def softmax(scores):
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# apply f(x) = Wx + b to all test images: prediction = class with highest score (argmax)
scores_all = scores_fn(W, b, X_all)
pred_all = scores_all.argmax(axis=1)
correct_all = pred_all == y_all

# selected examples: first test image of each class
sel = np.array([np.where(y_all == c)[0][0] for c in range(C)])
y_sel, scores_sel, pred_sel = y_all[sel], scores_all[sel], pred_all[sel]
imgs_sel = [to_image(X_raw[i]) for i in sel]

assert np.allclose(scores_sel[0], W @ X_all[sel[0]] + b)  # batched == single-image W @ x + b

# compute the hinge loss of the selected examples
# compute softmax probabilities of the selected examples and verify that each image sums to 1
hinge_i = hinge_loss(scores_sel, y_sel)
probs_sel = softmax(scores_sel)
prob_sums = probs_sel.sum(axis=1)
assert np.allclose(prob_sums, 1.0)
assert np.allclose(softmax(scores_all).sum(axis=1), 1.0)  # also for all test images


# Plots
MONO = {"family": "monospace"}

def sg(v):
    return f"{0.0 if abs(v) < 5e-4 else v:+.3f}"

def pl(v):
    return f"{0.0 if abs(v) < 5e-4 else v:.3f}"

def score_table_figure(imgs, labels, preds, vals, path, title, value_fmt):
    n = len(imgs)
    cw, lm, rh, img_s = 1.15, 1.5, 0.34, 0.95
    Wd = lm + cw * n + 0.2
    Hd = 0.55 + img_s + 0.35 + C * rh + 0.55
    fig = plt.figure(figsize=(Wd, Hd))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, Wd)
    ax.set_ylim(0, Hd)
    ax.axis("off")
    ax.text(0.15, Hd - 0.3, title, fontsize=12, fontweight="bold", va="center")
    y_top = Hd - 0.55
    for k in range(n):
        cx = lm + cw * k + cw / 2
        iax = fig.add_axes([(cx - img_s / 2) / Wd, (y_top - img_s) / Hd, img_s / Wd, img_s / Hd])
        iax.imshow(imgs[k])
        iax.set_xticks([])
        iax.set_yticks([])
        ax.text(cx, y_top - img_s - 0.17, f"true: {CLASS_NAMES[labels[k]]}", ha="center", fontsize=8)
    y0 = y_top - img_s - 0.35
    for c in range(C):
        yc = y0 - (c + 0.5) * rh
        ax.text(lm - 0.1, yc, CLASS_NAMES[c], ha="right", va="center", fontsize=9, fontweight="bold")
        for k in range(n):
            cx = lm + cw * k + cw / 2
            if preds[k] == c:
                col = "#cfe8cf" if preds[k] == labels[k] else "#f6d0cb"
                ax.add_patch(Rectangle((cx - cw / 2 + 0.05, yc - rh / 2 + 0.02), cw - 0.1, rh - 0.04,
                                       color=col, zorder=0))
            ax.text(cx, yc, value_fmt(vals[k, c]), ha="center", va="center", fontsize=9,
                    fontweight="bold" if labels[k] == c else "normal", **MONO)
    ax.text(0.15, 0.3, "bold = true class  |  shaded = highest score (prediction): green = correct, red = wrong",
            fontsize=7.5, color="#555")
    fig.savefig(path, dpi=150)
    plt.close(fig)

# 01 scores table: DISPLAY the scores of all 10 classes + predicted class for the selected example images
F_SCORES = "01_scores_table.png"
score_table_figure(imgs_sel, y_sel, pred_sel, scores_sel, IMG_DIR / F_SCORES,
                   "Class scores  f(x) = Wx + b  (random W, b = 0)", pl)

# 02 hinge loss per example: plot of the hinge loss per selected example and the AVERAGE loss
F_HINGE = "02_hinge_loss.png"
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.bar(range(C), hinge_i, color="#4c78a8")
ax.axhline(hinge_i.mean(), color="#c0392b", ls="--", lw=1.4, label=f"average = {hinge_i.mean():.3f}")
for k, v in enumerate(hinge_i):
    ax.text(k, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(C))
ax.set_xticklabels([f"#{k}\n{CLASS_NAMES[y_sel[k]]}" for k in range(C)], fontsize=8)
ax.set_ylabel("SVM hinge loss $L_i$")
ax.set_title("Multiclass SVM loss per selected example")
ax.set_ylim(0, hinge_i.max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(IMG_DIR / F_HINGE, dpi=150)
plt.close(fig)

# 03 softmax: raw scores vs probabilities (4 examples): COMPARE original scores with the resulting probabilities
F_SOFT = "03_scores_vs_probabilities.png"
show = [0, 3, 6, 9]
fig, axes = plt.subplots(len(show), 2, figsize=(10, 2.6 * len(show)), sharex=True)
for r, k in enumerate(show):
    for cidx, (vals, name) in enumerate([(scores_sel[k], "raw scores"), (probs_sel[k], "softmax probabilities")]):
        a = axes[r, cidx]
        colors = ["#4c78a8"] * C
        colors[y_sel[k]] = "#e08a2c"
        a.bar(range(C), vals, color=colors)
        a.axhline(0, color="k", lw=0.6)
        if cidx == 1:
            a.axhline(1 / C, color="#888", ls=":", lw=1)
        a.set_title(f"#{k} true: {CLASS_NAMES[y_sel[k]]}, pred: {CLASS_NAMES[pred_sel[k]]} - {name}", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
for a in axes[-1]:
    a.set_xticks(range(C))
    a.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
fig.suptitle("Scores vs. probabilities (orange = true class, dotted = uniform 0.1)", fontsize=11)
fig.tight_layout()
fig.savefig(IMG_DIR / F_SOFT, dpi=150)
plt.close(fig)

# 04 correct / misclassified: show correctly classified and misclassified examples (searched over the whole test set)
N_SHOW = 5
ok = np.where(correct_all)[0][:N_SHOW]
bad = np.where(~correct_all)[0][:N_SHOW]
F_EX = "04_correct_and_misclassified.png"
fig, axes = plt.subplots(2, N_SHOW, figsize=(2.3 * N_SHOW, 5.3))
for row, (idxs, name, col) in enumerate([(ok, "correct", "#2e8b57"), (bad, "misclassified", "#c0392b")]):
    for j in range(N_SHOW):
        a = axes[row, j]
        a.set_xticks([])
        a.set_yticks([])
        if j < len(idxs):
            i = idxs[j]
            a.imshow(to_image(X_raw[i]))
            a.set_title(f"true: {CLASS_NAMES[y_all[i]]}\npred: {CLASS_NAMES[pred_all[i]]}", fontsize=9, color=col)
            for sp in a.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(3)
        else:
            a.axis("off")
    axes[row, 0].set_ylabel(name, fontsize=11, color=col)
fig.tight_layout()
fig.savefig(IMG_DIR / F_EX, dpi=150)
plt.close(fig)

# 05 weights as 32x32x3 images: visualize the weight vector of each class as a 32 x 32 x 3 image
F_W = "05_weight_templates.png"
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for c, a in enumerate(axes.ravel()):
    w = W[c].reshape(3, 32, 32).transpose(1, 2, 0)
    a.imshow((w - w.min()) / (w.max() - w.min()))
    a.set_title(CLASS_NAMES[c])
    a.axis("off")
fig.suptitle("Rows of W as 32x32x3 images (random init, min-max scaled for display)", fontsize=11)
fig.tight_layout()
fig.savefig(IMG_DIR / F_W, dpi=150)
plt.close(fig)

# 06 histogram of the weights
F_WS = "06_weight_histogram.png"
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(W.ravel(), bins=80, color="#4c78a8")
ax.set_title("Histogram of all weights in W")
ax.set_xlabel("weight value")
ax.set_ylabel("count")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(IMG_DIR / F_WS, dpi=150)
plt.close(fig)


# Markdown report
def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


rel = IMG_DIR.name
n_ok = int((pred_sel == y_sel).sum())
md = ["# Linear Classification on CIFAR-10 (random weights, no training)\n"]

# report: parameters (shape, distribution, seed)
md.append("## 1. Parameters\n")
md.append(f"- Data: CIFAR-10 test batch, pixels scaled to [-0.5, 0.5], flattened to 3072 values.\n"
          f"- Selected examples: first test image of each class (indices `{sel.tolist()}`).\n"
          f"- `W`: shape {W.shape}, `0.01 * N(0,1)`, seed {SEED} (mean {W.mean():+.5f}, std {W.std():.5f}).\n"
          f"- `b`: shape {b.shape}, all zeros.\n")

# report: score tables + predictions
md.append("## 2. Scores f(x) = Wx + b\n")
md.append(f"![scores]({rel}/{F_SCORES})\n")
rows = [[CLASS_NAMES[c]] + [f"**{sg(scores_sel[k, c])}**" if y_sel[k] == c else sg(scores_sel[k, c])
                            for k in range(C)] for c in range(C)]
md.append(md_table(["class"] + [f"#{k} ({CLASS_NAMES[y_sel[k]]})" for k in range(C)], rows) + "\n")
md.append(md_table(["image", "true", "predicted", "correct?"],
                   [[f"#{k}", CLASS_NAMES[y_sel[k]], CLASS_NAMES[pred_sel[k]],
                     "yes" if pred_sel[k] == y_sel[k] else "no"] for k in range(C)]) + "\n")
md.append(f"{n_ok} of 10 selected images are predicted correctly. The scores are small and similar for all classes "
          f"(std of all scores: {scores_all.std():.3f}), and the highest one is decided by random correlations "
          "between W and the pixels, not by image content.\n")

# report: per-image loss, AVERAGE loss, worked example
md.append("## 3. Multiclass SVM hinge loss\n")
md.append("`L_i = sum_{j != y_i} max(0, s_j - s_yi + 1)`\n")
md.append(f"![hinge]({rel}/{F_HINGE})\n")
md.append(md_table(["image", "true", "s_true", "L_i"],
                   [[f"#{k}", CLASS_NAMES[y_sel[k]], sg(scores_sel[k, y_sel[k]]), f"{hinge_i[k]:.4f}"]
                    for k in range(C)]) + "\n")
md.append(f"**Average loss over the 10 selected examples: {hinge_i.mean():.4f}**\n")
k0 = 0
wr = []
for c in range(C):
    if c != y_sel[k0]:
        raw = scores_sel[k0, c] - scores_sel[k0, y_sel[k0]] + 1
        wr.append([CLASS_NAMES[c], sg(scores_sel[k0, c]), f"{raw:+.3f}", f"{max(0, raw):.3f}"])
md.append(f"Step-by-step hinge loss calculation for image #{k0} (true = {CLASS_NAMES[y_sel[k0]]}, s_true = "
          f"{sg(scores_sel[k0, y_sel[k0]])}):\n")
md.append(md_table(["wrong class j", "s_j", "s_j - s_y + 1", "max(0, .)"], wr) + f"\n\nSum = {hinge_i[k0]:.4f} = L_{k0}.\n")
kmin, kmax = int(hinge_i.argmin()), int(hinge_i.argmax())
md.append("All scores are small and similar, so each of the 9 wrong classes contributes a margin of roughly +1 and "
          "the loss is near C - 1 = 9 before any training (a useful sanity check). It varies around 9 because the "
          "margins depend on the true-class score: the lower s_true, the larger the loss. Here image "
          f"#{kmin} ({CLASS_NAMES[y_sel[kmin]]}, s_true = {sg(scores_sel[kmin, y_sel[kmin]])}) has the lowest loss "
          f"({hinge_i[kmin]:.2f}) and image #{kmax} ({CLASS_NAMES[y_sel[kmax]]}, s_true = "
          f"{sg(scores_sel[kmax, y_sel[kmax]])}) the highest ({hinge_i[kmax]:.2f}).\n")

# report: probabilities, table with sum = 1 per image
md.append("## 4. Softmax\n")
md.append(f"![softmax]({rel}/{F_SOFT})\n")
md.append(md_table(["image", "true", "predicted", "sum of probabilities", "max probability"],
                   [[f"#{k}", CLASS_NAMES[y_sel[k]], CLASS_NAMES[pred_sel[k]], f"{prob_sums[k]:.10f}",
                     f"{probs_sel[k].max():.4f}"] for k in range(C)]) + "\n")
md.append(f"All probability sums equal 1 (`np.allclose`: {bool(np.allclose(prob_sums, 1.0))}, also for all "
          f"{len(y_all)} test images). Softmax exponentiates the scores (making them positive) and divides by their "
          "sum. The class with the highest score keeps the highest probability. "
          f"Because the scores are nearly equal, the probabilities are close to uniform (about 0.1, largest "
          f"{probs_sel.max():.3f}).\n")

# report: explanation of correct vs. misclassified results
md.append("## 5. Correct and misclassified examples\n")
md.append(f"![examples]({rel}/{F_EX})\n")
md.append(f"Examples are taken from all {len(y_all)} test images. Overall the random classifier is correct for "
          f"{correct_all.mean() * 100:.1f}% of them, about chance level (10%). Correct predictions are coincidences: "
          "a random template happened to give the true class the highest score. Misclassified images show no "
          "pattern, because the weights contain no knowledge about the classes.\n")

# report: explanation why random weights show no learned class patterns
md.append("## 6. Weight vectors as 32x32x3 images\n")
md.append(f"![weights]({rel}/{F_W})\n")
md.append(f"![weight stats]({rel}/{F_WS})\n")
md.append("Each row of W is reshaped to 32x32x3 and min-max scaled for display. The weights were drawn "
          "independently from the same Gaussian, so every pixel weight is noise: no shapes, "
          "no colour layout, and no difference between the classes. Learned weights look like class templates only "
          "because training adjusts them using labelled images. Here the data never influenced W.\n")

OUT_FILE.write_text("\n".join(md), encoding="utf-8")
print(f"Wrote {OUT_FILE}\nPlots in {IMG_DIR}\naverage hinge loss = {hinge_i.mean():.4f}")