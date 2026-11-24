#!/usr/bin/env python3
"""
RQ1 experiment on the labelled Sagbas (2026) dataset.

RQ1: Does a model using all OBD signals classify driving style more accurately
     than the published speed-only reference?

What is compared (all on the same 5 s windows):
  A. Reference rule, no training  : Ameen et al. (2021) thresholds on OBD acceleration
  B. Unsupervised (Mohammadnazar) : k-means on volatility features, clusters named by volatility
  C. Supervised ablation          : Random Forest with feature groups added one at a time
                                    speed -> +engine -> +motion (friction circle) -> +context (Power Model)
  D. Leakage demonstration        : the same model with a naive random split
  E. Six-class task               : can speed alone separate eco vs non-eco (labels 4 vs 5)?
Plus convergent validity: harsh events, friction use and Power-Model risk per class.

Usage:
    python experiments/rq1_sagbas.py --data <folder with the extracted CSVs> \
        --out ../Documentation/Research/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intelligence import features as F                     # noqa: E402
from intelligence.classifier import (CLASSES_3, AmeenRule, VolatilityClusters,  # noqa: E402
                                     cluster_agreement, grouped_cv_predict, metrics,
                                     random_cv_predict)
from intelligence.datasets import sagbas                    # noqa: E402
from intelligence.events import detect_events               # noqa: E402

SEEDS = range(3)
GROUP_SETS = {
    "speed only (reference view)": (F.SPEED,),
    "speed + engine (all OBD)": (F.SPEED, F.ENGINE),
    "OBD + motion (friction circle)": (F.SPEED, F.ENGINE, F.MOTION),
    "OBD + motion + context (Power Model)": (F.SPEED, F.ENGINE, F.MOTION, F.CONTEXT),
}


def build_table(windows) -> pd.DataFrame:
    rows = []
    for w in windows:
        f = F.window_features(w.obd, w.motion)
        ev = detect_events(w.obd)                 # OBD-only events, the reference view
        f.update({
            "file": w.file, "label": w.label, "session": w.session,
            "n_harsh_events": sum(e.type != "overspeed" for e in ev),
        })
        rows.append(f)
    return pd.DataFrame(rows).fillna(0.0)


def cols_for(table: pd.DataFrame, groups) -> list[str]:
    probe = F.window_features(
        pd.DataFrame({"t": [0.0, 1.0, 2.0], "speed_kmh": [10.0, 11, 12], "rpm": [1, 1, 1],
                      "engine_load": [1, 1, 1], "maf": [1, 1, 1], "map": [1, 1, 1],
                      "speed_limit_kmh": [50, 50, 50]}),
        pd.DataFrame({"t": [0.0, 0.5, 1.0], "a_h_g": [0.1, 0.1, 0.1], "a_lat_g": [0, 0, 0],
                      "yaw_deg_s": [0, 0, 0], "mu": [0.8, 0.8, 0.8]}),
        groups=groups)
    return [c for c in probe if c in table.columns]


def pipeline_columns(table: pd.DataFrame) -> list[str]:
    """Speed + engine features WITHOUT MAF/MAP: the simulator and the live
    pipeline publish speed, RPM and engine load, not MAF/MAP."""
    return [c for c in cols_for(table, (F.SPEED, F.ENGINE))
            if not c.startswith(("maf_", "map_"))]


def rf(seed=0):
    return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  min_samples_leaf=2, n_jobs=-1, random_state=seed)


def hgb(seed=0):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          class_weight="balanced", random_state=seed)


def repeated(model_factory, X, y, groups, labels):
    runs = [metrics(y, grouped_cv_predict(model_factory, X, y, groups, seed=s), labels)
            for s in SEEDS]
    acc = np.array([r["accuracy"] for r in runs])
    f1 = np.array([r["macro_f1"] for r in runs])
    return {"accuracy_mean": acc.mean(), "accuracy_sd": acc.std(),
            "macro_f1_mean": f1.mean(), "macro_f1_sd": f1.std(),
            "confusion_seed0": runs[0]["confusion"], "labels": list(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="CSV to cache the feature table")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.cache and os.path.exists(args.cache):
        table = pd.read_csv(args.cache)
    else:
        table = build_table(sagbas.load(args.data))
        if args.cache:
            table.to_csv(args.cache, index=False)

    res: dict = {"n_windows": int(len(table)),
                 "n_sessions": int(table["session"].nunique()),
                 "windows_per_label": table["label"].value_counts().sort_index().to_dict(),
                 "sessions_per_label": table.groupby("label")["session"].nunique().to_dict()}

    # ---------------- three-class safety task ----------------
    t3 = table[table["label"].isin([1, 2, 3])].reset_index(drop=True)
    y3 = t3["label"].map(sagbas.SAFETY_3CLASS).to_numpy()
    g3 = t3["session"].to_numpy()

    res["A_reference_ameen"] = metrics(y3, AmeenRule().predict(t3), CLASSES_3)

    vc = VolatilityClusters(k=3).fit(t3)
    pred_c = vc.predict(t3)
    res["B_volatility_clusters"] = {
        **metrics(y3, pred_c, CLASSES_3), **cluster_agreement(y3, pred_c),
        "silhouette_k3": vc.silhouette(t3),
        "silhouette_by_k": {k: VolatilityClusters(k=k).fit(t3).silhouette(t3) for k in (2, 3, 4, 5)},
    }
    vci = VolatilityClusters(k=3, columns=F.VOLATILITY_COLUMNS_IMU).fit(t3)
    pred_ci = vci.predict(t3)
    res["B2_volatility_clusters_imu"] = {
        **metrics(y3, pred_ci, CLASSES_3), **cluster_agreement(y3, pred_ci),
        "silhouette_k3": vci.silhouette(t3),
    }

    res["C_supervised_ablation"] = {
        name: repeated(rf, t3[cols_for(t3, gs)], y3, g3, CLASSES_3)
        for name, gs in GROUP_SETS.items()
    }
    pipe_cols = pipeline_columns(t3)
    res["C_supervised_ablation"]["live-pipeline set (speed + RPM + load)"] = \
        repeated(rf, t3[pipe_cols], y3, g3, CLASSES_3)
    full_cols = cols_for(t3, GROUP_SETS["OBD + motion + context (Power Model)"])
    res["C_gradient_boosting_full"] = repeated(hgb, t3[full_cols], y3, g3, CLASSES_3)

    leak = metrics(y3, random_cv_predict(rf, t3[full_cols], y3), CLASSES_3)
    res["D_leakage_random_split"] = leak

    model = rf().fit(t3[full_cols], y3)
    imp = pd.Series(model.feature_importances_, index=full_cols).sort_values(ascending=False)
    res["feature_importance_top12"] = imp.head(12).round(4).to_dict()

    validity = t3.assign(cls=y3).groupby("cls").agg(
        windows=("label", "size"),
        share_with_harsh_event=("n_harsh_events", lambda s: float((s > 0).mean())),
        friction_u_max_mean=("friction_u_max", "mean"),
        a_h_g_max_mean=("a_h_g_max", "mean"),
        obd_a_abs_max_ms2_mean=("obd_a_abs_max_ms2", "mean"),
        friction_u_max_median=("friction_u_max", "median"),
        power_rr_median=("power_rr", "median"),
        speed_ratio_max_median=("speed_ratio_max", "median"),
        vol_ah_cv_median=("vol_ah_cv", "median"),
    ).reindex(CLASSES_3)
    res["convergent_validity"] = validity.round(3).to_dict(orient="index")

    # ---------------- six-class task ----------------
    y6 = table["label"].map(sagbas.LABEL_NAMES).to_numpy()
    g6 = table["session"].to_numpy()
    labels6 = list(sagbas.LABEL_NAMES.values())
    res["E_six_class"] = {
        name: repeated(rf, table[cols_for(table, gs)], y6, g6, labels6)
        for name, gs in (("speed only", GROUP_SETS["speed only (reference view)"]),
                         ("speed + engine", GROUP_SETS["speed + engine (all OBD)"]),
                         ("all signals", GROUP_SETS["OBD + motion + context (Power Model)"]))
    }

    # ---------------- model for the live scoring service ----------------
    # Trained on every 3-class window with the features the pipeline has.
    models_dir = Path(__file__).resolve().parents[1] / "intelligence" / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump({"model": rf().fit(t3[pipe_cols], y3), "columns": pipe_cols,
                 "trained_on": "Sagbas (2026) labels 1-3, 5 s windows",
                 "loo_drive_macro_f1": res["C_supervised_ablation"][
                     "live-pipeline set (speed + RPM + load)"]["macro_f1_mean"]},
                models_dir / "pipeline_rf.joblib")

    with open(os.path.join(args.out, "rq1_sagbas_results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    write_markdown(res, os.path.join(args.out, "rq1_sagbas_results.md"))
    print(open(os.path.join(args.out, "rq1_sagbas_results.md")).read())


def _pct(x):
    return f"{100 * x:.1f}%"


def _cm(cm, labels):
    head = "| true \\ predicted | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n"
    return head + "".join(f"| {lab} | " + " | ".join(str(v) for v in row) + " |\n"
                          for lab, row in zip(labels, cm))


def write_markdown(r: dict, path: str):
    L = []
    L.append("# RQ1 results on the labelled Sağbaş (2026) dataset\n")
    L.append("Generated by `source code/experiments/rq1_sagbas.py`. Do not edit by hand.\n")
    L.append(f"- Windows: {r['n_windows']} (5 s each), drives (sessions): {r['n_sessions']}")
    L.append(f"- Windows per label: {r['windows_per_label']}")
    L.append(f"- Drives per label: {r['sessions_per_label']}")
    L.append("- Evaluation: **leave-one-drive-out** cross-validation. Every window is predicted by a "
             "model that never saw any part of its drive. Supervised rows show mean ± sd over 3 model "
             "seeds (the split itself is deterministic).\n")

    L.append("## Three-class task (calm / normal / aggressive)\n")
    L.append("| Method | Training | Accuracy | Macro-F1 |\n|---|---|---|---|")
    a = r["A_reference_ameen"]
    L.append(f"| A. Reference rule (Ameen 2021 thresholds on OBD accel) | none | {_pct(a['accuracy'])} | {_pct(a['macro_f1'])} |")
    b = r["B_volatility_clusters"]
    L.append(f"| B. Volatility k-means, OBD speed at 1 Hz (Mohammadnazar 2021) | unsupervised | {_pct(b['accuracy'])} | {_pct(b['macro_f1'])} |")
    b2 = r["B2_volatility_clusters_imu"]
    L.append(f"| B2. Volatility k-means, accelerometer at 50 Hz | unsupervised | {_pct(b2['accuracy'])} | {_pct(b2['macro_f1'])} |")
    for name, m in r["C_supervised_ablation"].items():
        L.append(f"| C. Random Forest: {name} | supervised | {_pct(m['accuracy_mean'])} ± {_pct(m['accuracy_sd'])} | "
                 f"{_pct(m['macro_f1_mean'])} ± {_pct(m['macro_f1_sd'])} |")
    g = r["C_gradient_boosting_full"]
    L.append(f"| C. Gradient boosting: all signals | supervised | {_pct(g['accuracy_mean'])} ± {_pct(g['accuracy_sd'])} | "
             f"{_pct(g['macro_f1_mean'])} ± {_pct(g['macro_f1_sd'])} |")
    d = r["D_leakage_random_split"]
    L.append(f"| D. RF all signals, **naive random split (leaky, do not report as result)** | supervised | "
             f"{_pct(d['accuracy'])} | {_pct(d['macro_f1'])} |\n")

    L.append(f"Volatility clusters: adjusted Rand index {b['adjusted_rand_index']:.3f}, "
             f"silhouette (k=3) {b['silhouette_k3']:.3f}; silhouette by k: "
             + ", ".join(f"k={k}: {v:.3f}" for k, v in b["silhouette_by_k"].items())
             + f". Accelerometer variant: ARI {b2['adjusted_rand_index']:.3f}, silhouette {b2['silhouette_k3']:.3f}.\n")

    L.append("### Confusion matrices\n")
    L.append("Reference rule (A):\n\n" + _cm(a["confusion"], a["labels"]))
    best = r["C_supervised_ablation"]["OBD + motion + context (Power Model)"]
    L.append("Random Forest, all signals (C, seed 0):\n\n" + _cm(best["confusion_seed0"], best["labels"]))

    L.append("### Most important features (RF, all signals, fit on all 3-class windows)\n")
    L.append("| Feature | Importance |\n|---|---|")
    L += [f"| `{k}` | {v:.3f} |" for k, v in r["feature_importance_top12"].items()]

    L.append("\n### Convergent validity: do the physics measures rise with the label?\n")
    cv = r["convergent_validity"]
    cols = list(next(iter(cv.values())).keys())
    L.append("| class | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for cls, row in cv.items():
        L.append(f"| {cls} | " + " | ".join(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |")

    L.append("\n## Six-class task (adds eco / non-eco constant-speed driving)\n")
    L.append("| Features | Accuracy | Macro-F1 |\n|---|---|---|")
    for name, m in r["E_six_class"].items():
        L.append(f"| {name} | {_pct(m['accuracy_mean'])} ± {_pct(m['accuracy_sd'])} | "
                 f"{_pct(m['macro_f1_mean'])} ± {_pct(m['macro_f1_sd'])} |")
    six = r["E_six_class"]["speed only"]
    L.append("\nSpeed only (seed 0):\n\n" + _cm(six["confusion_seed0"], six["labels"]))
    six = r["E_six_class"]["all signals"]
    L.append("All signals (seed 0):\n\n" + _cm(six["confusion_seed0"], six["labels"]))
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
