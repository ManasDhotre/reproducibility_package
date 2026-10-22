"""
Reference experiment for the confidence-normalisation step of a multi-agent diagnostic gateway.

Three public tabular medical datasets x three model families = nine "agents".
Each agent emits a native confidence in a different form (probability, vote fraction,
squashed SVM margin). A calibration split is used to learn a per-agent normaliser
(Platt scaling or isotonic regression). We compare calibration quality of the native
and normalised positive-class probability on a held-out test split, over repeated
stratified splits, and verify that routing through the gateway never changes the label.

Run:  python reference_experiment.py
Needs: numpy pandas scikit-learn scipy matplotlib ; pima.csv and heart.csv in the same folder.
"""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
N_REPEATS = 50
EPS = 1e-6


# ------------------------------------------------------------------ data
def load_datasets():
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Pedigree", "Age", "Outcome"]
    pima = pd.read_csv("pima.csv", header=None, names=cols)
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        pima[c] = pima[c].replace(0, np.nan)          # impossible zeros -> missing (imputed inside the pipeline)
    heart = pd.read_csv("heart.csv")
    bc = load_breast_cancer(as_frame=True)
    breast = bc.frame.copy()
    # in sklearn's copy 0 = malignant, 1 = benign; make "1" the disease class
    breast["target"] = 1 - breast["target"]
    return {
        "Diabetes (PIMA)": (pima.drop(columns="Outcome").values, pima["Outcome"].values),
        "Heart (UCI)": (heart.drop(columns="target").values, 1 - heart["target"].values if heart["target"].mean() > 0.5 else heart["target"].values),
        "Breast (WDBC)": (breast.drop(columns="target").values, breast["target"].values),
    }


# ------------------------------------------------------------------ agents
def make_model(kind, seed):
    if kind == "LogReg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "RandForest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=200, random_state=seed))
    if kind == "LinearSVM":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearSVC(C=1.0, max_iter=20000, random_state=seed))
    raise ValueError(kind)


def native_score(model, X, kind):
    """Native confidence that the case is positive, in the form each family emits it."""
    if kind == "LinearSVM":
        m = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-m))        # naive squashing of the raw margin
    return model.predict_proba(X)[:, 1]        # LogReg probability / forest vote fraction


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Normaliser:
    """Per-agent confidence normaliser fitted on a calibration split."""
    def __init__(self, method):
        self.method = method

    def fit(self, s, y):
        if self.method == "platt":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s).reshape(-1, 1), y)
        else:
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
        return self

    def transform(self, s):
        if self.method == "platt":
            return self.m.predict_proba(logit(s).reshape(-1, 1))[:, 1]
        return self.m.predict(s)


class Agent:
    def __init__(self, name, kind, seed):
        self.name, self.kind, self.model, self.norm = name, kind, make_model(kind, seed), None

    def fit(self, Xtr, ytr, Xcal, ycal, method):
        self.model.fit(Xtr, ytr)
        self.norm = Normaliser(method).fit(native_score(self.model, Xcal, self.kind), ycal)
        return self

    def predict(self, X):
        raw = native_score(self.model, X, self.kind)
        return self.model.predict(X), raw, self.norm.transform(raw)


class Gateway:
    """Routes a request to an agent and returns label + normalised confidence."""
    def __init__(self, agents):
        self.agents = {a.name: a for a in agents}

    def run(self, agent_name, X):
        return self.agents[agent_name].predict(X)


# ------------------------------------------------------------------ metrics
def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return out


def run():
    data = load_datasets()
    kinds = ["LogReg", "RandForest", "LinearSVM"]
    rows, agree, sizes = [], [], {}
    for dname, (X, y) in data.items():
        sizes[dname] = {"n": int(len(y)), "pos": int(y.sum()), "pos_rate": float(y.mean()), "features": int(X.shape[1])}
        for kind in kinds:
            rec = {m: {"acc": [], "brier_raw": [], "brier_n": [], "ece_raw": [], "ece_n": [], "gap_raw": [], "gap_n": []} for m in ("platt", "isotonic")}
            for r in range(N_REPEATS):
                Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1000 + r)
                Xtr, Xcal, ytr, ycal = train_test_split(Xtv, ytv, test_size=0.25, stratify=ytv, random_state=2000 + r)
                for method in ("platt", "isotonic"):
                    ag = Agent(f"{dname}|{kind}", kind, r).fit(Xtr, ytr, Xcal, ycal, method)
                    gw = Gateway([ag])
                    label, raw, nrm = gw.run(ag.name, Xte)
                    standalone = ag.model.predict(Xte)
                    agree.append(float(np.mean(label == standalone)))
                    acc = accuracy_score(yte, label)
                    # confidence assigned to the predicted class
                    c_raw = np.where(label == 1, raw, 1 - raw)
                    c_nrm = np.where(label == 1, nrm, 1 - nrm)
                    d = rec[method]
                    d["acc"].append(acc)
                    d["brier_raw"].append(brier_score_loss(yte, raw)); d["brier_n"].append(brier_score_loss(yte, nrm))
                    d["ece_raw"].append(ece(raw, yte)); d["ece_n"].append(ece(nrm, yte))
                    d["gap_raw"].append(abs(c_raw.mean() - acc)); d["gap_n"].append(abs(c_nrm.mean() - acc))
            for method in ("platt", "isotonic"):
                d = rec[method]
                w = stats.wilcoxon(d["ece_raw"], d["ece_n"])
                rows.append({
                    "dataset": dname, "model": kind, "method": method,
                    "acc_mean": np.mean(d["acc"]), "acc_sd": np.std(d["acc"], ddof=1),
                    "ece_raw_mean": np.mean(d["ece_raw"]), "ece_raw_sd": np.std(d["ece_raw"], ddof=1),
                    "ece_n_mean": np.mean(d["ece_n"]), "ece_n_sd": np.std(d["ece_n"], ddof=1),
                    "brier_raw_mean": np.mean(d["brier_raw"]), "brier_n_mean": np.mean(d["brier_n"]),
                    "gap_raw_mean": np.mean(d["gap_raw"]), "gap_n_mean": np.mean(d["gap_n"]),
                    "wilcoxon_p": float(w.pvalue),
                    "ece_better_frac": float(np.mean(np.array(d["ece_n"]) < np.array(d["ece_raw"]))),
                })
    df = pd.DataFrame(rows)
    # cross-agent comparability: spread of confidence-accuracy gap across the nine agents
    out = {"sizes": sizes, "agreement_min": float(np.min(agree)), "agreement_mean": float(np.mean(agree)), "n_gateway_checks": len(agree)}
    for method in ("platt", "isotonic"):
        s = df[df.method == method]
        out[method] = {
            "ece_raw_overall": float(s.ece_raw_mean.mean()), "ece_n_overall": float(s.ece_n_mean.mean()),
            "gap_raw_overall": float(s.gap_raw_mean.mean()), "gap_n_overall": float(s.gap_n_mean.mean()),
            "gap_raw_spread": float(s.gap_raw_mean.std(ddof=1)), "gap_n_spread": float(s.gap_n_mean.std(ddof=1)),
            "configs_improved": int((s.ece_n_mean < s.ece_raw_mean).sum()), "configs": int(len(s)),
        }
    df.to_csv("results_table.csv", index=False)
    json.dump(out, open("results_summary.json", "w"), indent=2)
    print(df.round(4).to_string())
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
