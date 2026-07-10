"""Judge whether to keep or discard an experiment run.

Compares the current run's metrics against all previously kept runs in
results.tsv. A recall floor (90% of best kept recall) prevents degenerate
"return everything" strategies. Above that floor, IoU is the decider.

Usage:
    python judge.py --iou 0.0193 --precision_omega 0.1325 --recall 0.4800 --precision 0.0194 --avg_chars 10971
"""

import argparse
import os
import sys

import pandas as pd

TSV_PATH = "results.tsv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou", type=float, required=True)
    parser.add_argument("--precision_omega", type=float, required=True)
    parser.add_argument("--recall", type=float, required=True)
    parser.add_argument("--precision", type=float, required=True)
    parser.add_argument("--avg_chars", type=float, required=True)
    args = parser.parse_args()

    new = {
        "iou": args.iou,
        "precision_omega": args.precision_omega,
        "recall": args.recall,
        "precision": args.precision,
        "avg_chars": args.avg_chars,
    }

    if not os.path.exists(TSV_PATH):
        print("No results.tsv yet — first run, always KEEP")
        sys.exit(0)

    df = pd.read_csv(TSV_PATH, sep="\t")
    kept = df[df["status"] == "keep"]

    if kept.empty:
        print("No kept runs yet — KEEP (first run)")
        sys.exit(0)

    best_recall = kept["recall"].max()
    best_iou = kept["iou"].max()

    recall_floor = 0.9 * best_recall
    recall_ok = new["recall"] >= recall_floor

    print(f"Best kept recall: {best_recall:.4f}  (floor: {recall_floor:.4f})")
    print(f"Best kept IoU:    {best_iou:.4f}")
    print(f"Current recall:   {new['recall']:.4f}  {'OK' if recall_ok else 'BELOW FLOOR'}")
    print(f"Current IoU:      {new['iou']:.4f}")
    print(f"Current Precision-Omega: {new['precision_omega']:.4f}")
    print(f"Current Precision:      {new['precision']:.4f}")
    print(f"Current Avg chars:      {new['avg_chars']:.0f}")

    if not recall_ok:
        print("DISCARD (recall below floor)")
        sys.exit(1)

    if new["iou"] > best_iou:
        print("KEEP (new best IoU)")
        sys.exit(0)
    else:
        print("DISCARD (IoU not improved)")
        sys.exit(1)


if __name__ == "__main__":
    main()
