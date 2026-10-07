"""Steps 3 & 6 - Evaluation and tuning with Character Error Rate (CER) and Word Error Rate (WER).

Dataset layout (see README):
    dataset/images/sample01.png   handwritten image
    dataset/labels/sample01.txt   correct typed text (ground truth)

Usage:
    python evaluate.py                    # page/paragraph images (line segmentation ON)
    python evaluate.py --single_line      # every image is ONE text line (e.g. IAM line images)
"""
import argparse, glob, json, os, time
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ocr_engine import HandwritingOCR, load_config


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def norm(t):
    return " ".join(t.lower().split())


def cer(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    return edit_distance(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    return edit_distance(r, h) / max(len(r), 1)


def load_dataset(root="dataset"):
    data = []
    for img_path in sorted(glob.glob(f"{root}/images/*")):
        name = os.path.splitext(os.path.basename(img_path))[0]
        lab = f"{root}/labels/{name}.txt"
        img = cv2.imread(img_path)
        if img is not None and os.path.exists(lab):
            data.append((name, img, open(lab, encoding="utf-8").read()))
    if not data:
        raise SystemExit("No image/label pairs found in dataset/. See README.")
    return data


def run(cfg, data, single_line):
    ocr = HandwritingOCR(cfg)
    rows = []
    for name, img, truth in data:
        res = ocr.transcribe_page(img, single_line=single_line)
        rows.append({"sample": name, "CER": cer(truth, res["text"]),
                     "WER": wer(truth, res["text"]), "seconds": res["seconds"]})
    df = pd.DataFrame(rows)
    return ocr, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_line", action="store_true")
    args = ap.parse_args()

    base = load_config()
    data = load_dataset()
    print(f"Loaded {len(data)} labelled samples")

    # Hyper-parameter search (Step 6: adjust settings from evaluation results)
    grid = [{"num_beams": b, "binarize": z} for b in (1, 3, 5) for z in (False, True)]
    summary, best, best_ocr = [], None, None
    for g in grid:
        cfg = {**base, **g}
        ocr, df = run(cfg, data, args.single_line)
        row = {**g, "mean_CER": df.CER.mean(), "mean_WER": df.WER.mean(),
               "mean_seconds": df.seconds.mean()}
        summary.append(row)
        print(row)
        if best is None or row["mean_CER"] < best["mean_CER"]:
            best, best_ocr, best_df = row, ocr, df

    os.makedirs("results", exist_ok=True)
    res = pd.DataFrame(summary)
    res.to_csv("results/tuning_results.csv", index=False)
    best_df.to_csv("results/best_per_sample.csv", index=False)

    labels = [f"beams={r.num_beams}\nbin={r.binarize}" for r in res.itertuples()]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, res.mean_CER, label="CER"); plt.plot(labels, res.mean_WER, "ro-", label="WER")
    plt.ylabel("Error rate (lower is better)"); plt.title("Tuning results"); plt.legend()
    plt.tight_layout(); plt.savefig("results/tuning_chart.png", dpi=150)

    # Save best configuration + model (reproducibility)
    final = {**base, "num_beams": int(best["num_beams"]), "binarize": bool(best["binarize"])}
    json.dump(final, open("config.json", "w"), indent=2)
    best_ocr.cfg = final
    best_ocr.save()
    print("\nBEST:", best)
    print("Saved best settings to config.json and model to saved_model/")
