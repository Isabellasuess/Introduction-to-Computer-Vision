import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# Load CIFAR-10
DATA_DIR = Path(__file__).resolve().parent.parent / "cifar-10" / "cifar-10-python" / "cifar-10-batches-py"

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Could not find data directory: {DATA_DIR}")

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
np.random.seed(0)
idx = np.random.permutation(len(X_train))[:5000]
X_train, y_train = X_train[idx], y_train[idx]
X_test, y_test = X_test[:1000], y_test[:1000]


# KNN classifier
class KNN:
    def __init__(self, k=1, metric="L1"):
        self.k = k
        self.metric = metric

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            if self.metric == "L1":
                # L1 (Manhattan)
                distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            elif self.metric == "L2":
                # L2 (Euclidean)
                distances = np.sqrt(np.sum((self.Xtr - X[i, :]) ** 2, axis=1))
            else:
                raise ValueError("metric must be 'L1' or 'L2'")

            nearest = np.argsort(distances)[:self.k]
            Ypred[i] = np.bincount(self.ytr[nearest]).argmax()

        return Ypred


# 5-fold cross-validation (choose k for a given metric)
def cross_validation(X, y, k, metric, folds=5):
    X_folds = np.array_split(X, folds)
    y_folds = np.array_split(y, folds)
    accs = []
    for f in range(folds):
        X_val, y_val = X_folds[f], y_folds[f]
        X_tr = np.concatenate(X_folds[:f] + X_folds[f + 1:])
        y_tr = np.concatenate(y_folds[:f] + y_folds[f + 1:])
        model = KNN(k, metric)
        model.train(X_tr, y_tr)
        accs.append(accuracy_score(y_val, model.predict(X_val)))
    return np.mean(accs)


K_VALUES = [1, 5, 9, 13, 15, 17]
METRICS = ["L1", "L2"]

for metric in METRICS:
    print("=" * 40)
    print("5-fold CV accuracy (%s distance)" % metric)
    best_acc, best_k = 0, None
    for k in K_VALUES:
        acc = cross_validation(X_train, y_train, k, metric)
        print("k=%d  accuracy=%.4f" % (k, acc))
        if acc > best_acc:
            best_acc, best_k = acc, k

    print("\nBest k = %d (CV accuracy %.4f)" % (best_k, best_acc))

    # Final evaluation on the test set with the best k for this metric
    model = KNN(best_k, metric)
    model.train(X_train, y_train)
    pred = model.predict(X_test)

    print("\nTest results (k=%d, %s)" % (best_k, metric))
    print("Accuracy : %.4f" % accuracy_score(y_test, pred))
    print("Precision: %.4f" % precision_score(y_test, pred, average="macro", zero_division=0))
    print("Recall   : %.4f" % recall_score(y_test, pred, average="macro", zero_division=0))
    print("F1-score : %.4f" % f1_score(y_test, pred, average="macro", zero_division=0))
    print()
