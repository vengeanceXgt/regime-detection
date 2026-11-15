"""Run the full experiment and write results/ (metrics table + reliability plot).

    python run_experiment.py                    # synthetic data, several seeds
    python run_experiment.py --real ^NSEI       # optional: real index data (needs network)
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_prices, simulate_regimes
from src.metrics import accuracy, brier, ece, log_loss, reliability
from src.walkforward import run_walk_forward

MODELS = ["hmm", "gbm", "gbm_calibrated", "mlp", "mlp_calibrated"]


def evaluate(res: pd.DataFrame) -> dict:
    y = res["y"].values
    return {
        m: {
            "accuracy": accuracy(y, res[m].values),
            "log_loss": log_loss(y, res[m].values),
            "brier": brier(y, res[m].values),
            "ece": ece(y, res[m].values),
        }
        for m in MODELS
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="ticker for real data, e.g. ^NSEI")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs, pooled = [], []
    datasets = [("real", load_prices(args.real))] if args.real else [
        (f"seed{s}", simulate_regimes(seed=s)) for s in range(args.seeds)
    ]
    for name, df in datasets:
        res = run_walk_forward(df, seed=0)
        pooled.append(res)
        scores = evaluate(res)
        for m, s in scores.items():
            runs.append({"dataset": name, "model": m, **s})
        print(f"finished {name}", flush=True)

    table = pd.DataFrame(runs)
    summary = table.groupby("model")[["accuracy", "log_loss", "brier", "ece"]].agg(["mean", "std"]).round(4)
    table.to_csv(os.path.join(args.out, "metrics_by_run.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "metrics_summary.csv"))
    print("\n" + summary.to_string())

    allres = pd.concat(pooled)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for m in ["hmm", "gbm", "gbm_calibrated", "mlp_calibrated"]:
        rel = reliability(allres["y"].values, allres[m].values)
        ax.plot(rel[:, 0], rel[:, 1], marker="o", ms=3, label=m)
    ax.set_xlabel("predicted P(turbulent regime tomorrow)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability (walk-forward, pooled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "reliability.png"), dpi=150)


if __name__ == "__main__":
    main()
