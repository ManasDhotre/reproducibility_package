import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import reference_experiment as R
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

data = R.load_datasets()
# ---- reliability diagram: PIMA, LinearSVM, pooled over all repeats
X, y = data["Diabetes (PIMA)"]
raw_all, nrm_all, y_all = [], [], []
for r in range(R.N_REPEATS):
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1000 + r)
    Xtr, Xcal, ytr, ycal = train_test_split(Xtv, ytv, test_size=0.25, stratify=ytv, random_state=2000 + r)
    ag = R.Agent("a", "LinearSVM", r).fit(Xtr, ytr, Xcal, ycal, "platt")
    _, raw, nrm = ag.predict(Xte)
    raw_all.append(raw); nrm_all.append(nrm); y_all.append(yte)
raw_all, nrm_all, y_all = map(np.concatenate, (raw_all, nrm_all, y_all))
fig, ax = plt.subplots(figsize=(3.4, 3.2))
edges = np.linspace(0, 1, 11)
for p, lab, mk, ls in ((raw_all, "Native (squashed margin)", "o", "--"), (nrm_all, "Normalised (Platt)", "s", "-")):
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 15:
            xs.append(p[m].mean()); ys.append(y_all[m].mean())
    ax.plot(xs, ys, marker=mk, ls=ls, color="black" if mk == "s" else "#777777", ms=4, lw=1.2, label=lab)
ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
ax.set_xlabel("Predicted probability of diabetes"); ax.set_ylabel("Observed frequency")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig("fig_reliability.png", dpi=300); plt.close(fig)

# ---- ECE bars
df = pd.read_csv("results_table.csv")
pl = df[df.method == "platt"].reset_index(drop=True)
iso = df[df.method == "isotonic"].reset_index(drop=True)
labels = [f"{d.split(' ')[0]}\n{m}" for d, m in zip(pl.dataset, pl.model)]
x = np.arange(len(pl)); w = 0.27
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.bar(x - w, pl.ece_raw_mean, w, yerr=pl.ece_raw_sd, color="#9a9a9a", label="Native", capsize=1.5, error_kw={"lw": 0.7})
ax.bar(x, pl.ece_n_mean, w, yerr=pl.ece_n_sd, color="#333333", label="Platt", capsize=1.5, error_kw={"lw": 0.7})
ax.bar(x + w, iso.ece_n_mean, w, yerr=iso.ece_n_sd, color="white", edgecolor="black", hatch="///", label="Isotonic", capsize=1.5, error_kw={"lw": 0.7})
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("Expected calibration error")
ax.set_ylim(0, 0.225); ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig("fig_ece.png", dpi=300); plt.close(fig)
print("ok")
