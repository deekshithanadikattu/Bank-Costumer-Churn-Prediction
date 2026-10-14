"""
Bank Customer Churn: analysis and prediction.

What this script does
1. Explores which customer segments churn most (EDA) and saves charts to figures/.
2. Compares three classifiers with stratified 5-fold cross-validation.
3. Evaluates the best one on a held-out test set with churn-focused metrics.
4. Ranks test customers by churn risk and measures how many churners the
   top 20% riskiest customers contain (what a retention team would act on).
5. Saves a scored customer list (at_risk_customers.csv) and results.json.

Fixes over the original notebook
- No oversampling before the train/test split (that leaked copies of the same
  customers into the test set and produced a fake 98% accuracy).
- customer_id is not used as a feature.
- Scaling and encoding live inside a pipeline fit on training data only.
- Country is one-hot encoded instead of coded 0/1/2.
- Reports recall, precision, ROC-AUC and PR-AUC next to the "always stays" baseline.

Usage (in the folder with the CSV):
    python churn_analysis.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files; no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_FILE = "Bank Customer Churn Prediction.csv"
FIG_DIR = Path("figures")
RANDOM_STATE = 42
TOP_SHARE = 0.20  # share of customers a retention team can contact

NUMERIC = ["credit_score", "age", "tenure", "balance", "products_number",
           "estimated_salary"]
BINARY = ["credit_card", "active_member"]
CATEGORICAL = ["country", "gender"]

FEATURE_LABELS = {
    "credit_score": "Credit score", "age": "Age", "tenure": "Tenure (years)",
    "balance": "Balance", "products_number": "Number of products",
    "estimated_salary": "Estimated salary", "credit_card": "Has credit card",
    "active_member": "Active member", "country": "Country", "gender": "Gender",
}

# Chart styling: one blue for data, gray for reference lines and text.
BLUE, GRAY, INK, MUTED = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


# ---------------------------------------------------------------- 1. EDA
def segment_table(df):
    """Churn rate for each customer segment, sorted from highest risk."""
    d = df.copy()
    d["age_band"] = pd.cut(d["age"], [17, 29, 39, 49, 59, 120],
                           labels=["18-29", "30-39", "40-49", "50-59", "60+"])
    d["products"] = d["products_number"].astype(str) + " product(s)"
    d["activity"] = np.where(d["active_member"] == 1, "Active", "Inactive")
    d["balance_status"] = np.where(d["balance"] == 0, "Zero balance", "Has balance")
    rows = []
    for col, name in [("country", "Country"), ("gender", "Gender"),
                      ("age_band", "Age"), ("products", "Products"),
                      ("activity", "Activity"), ("balance_status", "Balance")]:
        g = d.groupby(col, observed=True)["churn"].agg(["count", "mean"])
        for seg, r in g.iterrows():
            rows.append({"dimension": name, "segment": str(seg),
                         "customers": int(r["count"]),
                         "churn_rate": round(float(r["mean"]), 4)})
    return pd.DataFrame(rows)


def plot_segments(seg, overall):
    dims = seg["dimension"].unique()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, dim in zip(axes.flat, dims):
        s = seg[seg["dimension"] == dim]
        bars = ax.bar(s["segment"], s["churn_rate"] * 100, color=BLUE, width=0.6)
        ax.axhline(overall * 100, color=GRAY, linestyle="--", linewidth=1.2)
        for b, v in zip(bars, s["churn_rate"]):
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v:.0%}",
                    ha="center", fontsize=8, color=INK)
        ax.set_title(f"By {dim.lower()}")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Churn rate (%)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Churn rate by customer segment", fontweight="bold", fontsize=13)
    fig.text(0.5, 0.935, f"Dashed line = overall churn rate ({overall:.0%})",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "churn_by_segment.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. Models
def build_models(pos_weight):
    pre_scaled = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(drop="first"), CATEGORICAL),
    ])
    pre_trees = ColumnTransformer([
        ("num", "passthrough", NUMERIC + BINARY),
        ("cat", OneHotEncoder(drop="first"), CATEGORICAL),
    ])
    return {
        "Logistic Regression": Pipeline([
            ("prep", pre_scaled),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("prep", pre_trees),
            ("clf", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("prep", pre_trees),
            ("clf", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def plot_model_comparison(cv):
    names = list(cv)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [cv[n]["pr_auc_mean"] for n in names]
    errs = [cv[n]["pr_auc_std"] for n in names]
    bars = ax.bar(x, vals, yerr=errs, color=BLUE, width=0.5,
                  error_kw={"ecolor": MUTED, "capsize": 4, "linewidth": 1})
    for b, v, e in zip(bars, vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.02, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC (5-fold CV, train set)")
    ax.set_title("Model comparison (higher is better; error bars = 1 std)")
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. Targeting
def gains_curve(y_true, scores):
    """Share of all churners captured when contacting the top-k riskiest customers."""
    order = np.argsort(-scores)
    captured = np.cumsum(np.asarray(y_true)[order]) / np.sum(y_true)
    contacted = np.arange(1, len(order) + 1) / len(order)
    return contacted, captured


def plot_gains(contacted, captured, top_capture):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(contacted * 100, captured * 100, color=BLUE, linewidth=2, label="Model ranking")
    ax.plot([0, 100], [0, 100], color=GRAY, linestyle="--", linewidth=1.2,
            label="Random contact")
    ax.scatter([TOP_SHARE * 100], [top_capture * 100], color=BLUE, s=50, zorder=3,
               edgecolor="#fcfcfb", linewidth=2)
    ax.annotate(f"Top {TOP_SHARE:.0%} of customers\ncontain {top_capture:.0%} of churners",
                (TOP_SHARE * 100, top_capture * 100), xytext=(32, top_capture * 100 - 22),
                fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xlabel("Customers contacted, riskiest first (%)")
    ax.set_ylabel("Churners captured (%)")
    ax.set_title("Cumulative gains on the held-out test set")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.grid(color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cumulative_gains.png", dpi=150)
    plt.close(fig)


def plot_importance(imp):
    imp = imp.sort_values("importance")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(imp["label"], imp["importance"], color=BLUE, height=0.6,
            xerr=imp["std"], error_kw={"ecolor": MUTED, "capsize": 3, "linewidth": 1})
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("What drives churn predictions (permutation importance)")
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    overall = df["churn"].mean()
    print(f"Loaded {len(df):,} customers; churn rate {overall:.1%}.")

    # 1. EDA
    seg = segment_table(df)
    plot_segments(seg, overall)
    print("\nHighest-churn segments:")
    print(seg.sort_values("churn_rate", ascending=False).head(8).to_string(index=False))
    germany_zero_balance = float((df.loc[df["country"] == "Germany", "balance"] == 0).mean())
    others_zero_balance = float((df.loc[df["country"] != "Germany", "balance"] == 0).mean())

    # 2. Split once; the test set is never touched until the end.
    X = df[NUMERIC + BINARY + CATEGORICAL]
    y = df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    ids_te = df.loc[X_te.index, "customer_id"]

    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    models = build_models(pos_weight)
    cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv = {}
    print("\n5-fold cross-validation on the training set:")
    for name, model in models.items():
        r = cross_validate(model, X_tr, y_tr, cv=cv_split,
                           scoring=["average_precision", "roc_auc", "recall", "precision"])
        cv[name] = {
            "pr_auc_mean": round(r["test_average_precision"].mean(), 4),
            "pr_auc_std": round(r["test_average_precision"].std(), 4),
            "roc_auc_mean": round(r["test_roc_auc"].mean(), 4),
            "recall_mean": round(r["test_recall"].mean(), 4),
            "precision_mean": round(r["test_precision"].mean(), 4),
        }
        print(f"  {name:20s} PR-AUC {cv[name]['pr_auc_mean']:.3f} "
              f"(±{cv[name]['pr_auc_std']:.3f})  ROC-AUC {cv[name]['roc_auc_mean']:.3f}  "
              f"recall {cv[name]['recall_mean']:.3f}  precision {cv[name]['precision_mean']:.3f}")
    plot_model_comparison(cv)

    best_name = max(cv, key=lambda n: cv[n]["pr_auc_mean"])
    best = models[best_name].fit(X_tr, y_tr)
    print(f"\nBest model by CV PR-AUC: {best_name}")

    # 3. Held-out test evaluation
    p_te = best.predict_proba(X_te)[:, 1]
    y_pred = (p_te >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    test = {
        "model": best_name,
        "customers": int(len(y_te)),
        "churners": int(y_te.sum()),
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "always_stays_baseline_accuracy": round(1 - y_te.mean(), 4),
        "recall": round(recall_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred), 4),
        "f1": round(f1_score(y_te, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_te, p_te), 4),
        "pr_auc": round(average_precision_score(y_te, p_te), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # 4. Targeting: top 20% riskiest customers
    contacted, captured = gains_curve(y_te.values, p_te)
    k = int(round(TOP_SHARE * len(y_te)))
    top_idx = np.argsort(-p_te)[:k]
    top_capture = float(y_te.values[top_idx].sum() / y_te.sum())
    top_churn_rate = float(y_te.values[top_idx].mean())
    test["top_20pct"] = {
        "customers_contacted": k,
        "share_of_churners_captured": round(top_capture, 4),
        "churn_rate_in_group": round(top_churn_rate, 4),
        "lift_vs_random": round(top_capture / TOP_SHARE, 2),
    }
    plot_gains(contacted, captured, top_capture)

    print("\n=== Held-out test set (report these) ===")
    for key, v in test.items():
        print(f"  {key:32s} {v}")

    # Feature importance on the test set
    pi = permutation_importance(best, X_te, y_te, scoring="average_precision",
                                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"feature": X_te.columns, "importance": pi.importances_mean,
                        "std": pi.importances_std})
    imp["label"] = imp["feature"].map(FEATURE_LABELS)
    plot_importance(imp)
    print("\nPermutation importance (drop in PR-AUC):")
    print(imp.sort_values("importance", ascending=False)[["label", "importance"]]
          .round(4).to_string(index=False))

    # 5. Scored customer list for a retention team (and for a Tableau dashboard)
    scored = df.loc[X_te.index].assign(customer_id=ids_te, churn_risk=p_te.round(4))
    scored["risk_tier"] = pd.qcut(scored["churn_risk"].rank(method="first", ascending=False),
                                  [0, TOP_SHARE, 0.5, 1.0],
                                  labels=["High (top 20%)", "Medium", "Low"])
    scored.sort_values("churn_risk", ascending=False).to_csv("at_risk_customers.csv", index=False)

    results = {
        "overall_churn_rate": round(float(overall), 4),
        "segments": seg.to_dict(orient="records"),
        "zero_balance_share": {"Germany": round(germany_zero_balance, 4),
                               "France_and_Spain": round(others_zero_balance, 4)},
        "cross_validation": cv,
        "test": test,
        "feature_importance": imp.sort_values("importance", ascending=False)
                                 [["feature", "importance"]].round(4).to_dict(orient="records"),
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved figures/, at_risk_customers.csv and results.json")


if __name__ == "__main__":
    main()
