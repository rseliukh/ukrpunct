#!/usr/bin/env python3
"""
metrics.py — evaluation for the Ukrainian punctuation & capitalization benchmark.

INPUT: a gold directory and a prediction directory with identical file names
(<segment_id>.tsv), each file in the format

    token<TAB>punct_tag<TAB>cap_tag      (first line is this header)

Prediction files must be aligned with gold token by token (same number of
rows, same word order). Aligning a model's raw output to the input tokens is
the model wrapper's job; this script only computes metrics. Segments whose
length differs from gold are excluded and reported.

COMPUTES:
  - precision / recall / F1 per punctuation class, class O excluded;
  - Overall micro P/R/F1 excluding O — the main metric, as in the punctuator2
    line of work (Tilk & Alumäe 2016; Sazhok et al. 2021);
  - capitalization on sentence-internal tokens only: tokens whose preceding
    gold token ends a sentence (PERIOD/QUESTION) and the first token of a
    segment are excluded, since capitalizing them is trivial. Reports recall
    (restored / true CAP) and precision (restored / (restored + false CAP));
    a false CAP right after a *predicted* sentence end is not counted, since
    it is a punctuation error rather than a capitalization one;
  - 95% bootstrap CI (1000 resamples by default) over SEGMENTS, not tokens,
    for Overall F1 and for capitalization recall/precision;
  - optionally appends a result row to a CSV log.

USAGE:
    python src/metrics.py --gold data/bruk/fiction-window/gold \\
        --pred my_preds/fiction-window --model my-model
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

TRACK_CLASSES = {
    "A": ["O", "COMMA", "PERIOD", "QUESTION"],
    "B": None,  # taken from the gold data (all classes that occur)
}

# Tags after which capitalization is trivial (start of a new sentence)
SENT_END_TAGS = {"PERIOD", "QUESTION"}


# --------------------------------------------------------------------------
# Loading TSV
# --------------------------------------------------------------------------

def read_tsv(path: Path):
    """Return (tokens, punct_tags, cap_tags) as lists, in row order."""
    tokens, punct, cap = [], [], []
    with path.open(encoding="utf-8") as f:
        f.readline()  # header: token\tpunct_tag\tcap_tag
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}: expected 3 columns, got {parts!r}")
            tok, p, c = parts
            tokens.append(tok)
            punct.append(p)
            cap.append(c)
    return tokens, punct, cap


def load_segments(gold_dir: Path, pred_dir: Path, exclude_segment_start=True):
    """
    Return (segments, missing_pred, extra_pred, length_mismatches).
    Each segment is a dict(segment_id, gold_punct, pred_punct, gold_cap,
    pred_cap, cap_eval_gold, cap_eval_pred, cap_false). Segments whose length
    differs from gold are not dropped silently: they are returned in
    length_mismatches so the scale of the alignment problem is visible.
    """
    gold_files = {p.stem: p for p in sorted(gold_dir.glob("*.tsv"))}
    pred_files = {p.stem: p for p in sorted(pred_dir.glob("*.tsv"))}

    missing_pred = sorted(set(gold_files) - set(pred_files))
    extra_pred = sorted(set(pred_files) - set(gold_files))

    segments = []
    length_mismatches = []

    for seg_id in sorted(set(gold_files) & set(pred_files)):
        g_tok, g_punct, g_cap = read_tsv(gold_files[seg_id])
        p_tok, p_punct, p_cap = read_tsv(pred_files[seg_id])
        if len(g_tok) != len(p_tok):
            length_mismatches.append((seg_id, len(g_tok), len(p_tok)))
            continue
        cap_eval_gold, cap_eval_pred, cap_false = mid_sentence_cap(
            g_punct, p_punct, g_cap, p_cap, exclude_segment_start)
        segments.append({
            "segment_id": seg_id,
            "gold_punct": g_punct,
            "pred_punct": p_punct,
            "gold_cap": g_cap,
            "pred_cap": p_cap,
            "cap_eval_gold": cap_eval_gold,
            "cap_eval_pred": cap_eval_pred,
            "cap_false": cap_false,
        })

    return segments, missing_pred, extra_pred, length_mismatches


def mid_sentence_cap(g_punct, p_punct, g_cap, p_cap, exclude_segment_start=True):
    """
    Capitalization evaluated inside sentences only. Positions whose preceding
    GOLD token has PERIOD/QUESTION are excluded entirely (gold, not pred, so
    the set of positions is the same for every model).
    Returns (gold_eval, pred_eval, n_false):
      gold_eval / pred_eval -- parallel lists for gold CAP tokens
        (gold_eval is all "CAP", pred_eval is what the model predicted);
      n_false -- false capitalization: gold=O, pred=CAP at the same positions,
        EXCEPT where the preceding token in the PREDICTION has PERIOD/QUESTION
        (a capital there follows from a wrong sentence boundary, which is a
        punctuation error, not a separate casing error).
    The first token of a segment is excluded by default (a segment starts a
    sentence); exclude_segment_start=False keeps it.
    """
    gold_eval, pred_eval = [], []
    n_false = 0
    for i, (gc, pc) in enumerate(zip(g_cap, p_cap)):
        if i == 0:
            if exclude_segment_start:
                continue
        elif g_punct[i - 1] in SENT_END_TAGS:
            continue
        if gc == "CAP":
            gold_eval.append(gc)
            pred_eval.append(pc)
        elif pc == "CAP":
            if i > 0 and p_punct[i - 1] in SENT_END_TAGS:
                continue
            n_false += 1
    return gold_eval, pred_eval, n_false


# --------------------------------------------------------------------------
# P/R/F1
# --------------------------------------------------------------------------

def prf1_per_class(gold, pred, classes):
    """
    gold, pred: parallel tag lists (concatenated over all segments).
    Returns dict: class -> (precision, recall, f1, support).
    """
    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter()

    for g, p in zip(gold, pred):
        support[g] += 1
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] += 1

    result = {}
    for c in classes:
        tp_c, fp_c, fn_c = tp[c], fp[c], fn[c]
        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[c] = (prec, rec, f1, support[c])
    return result


def accuracy(gold, pred):
    if not gold:
        return 0.0
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)


def micro_prf1_excl_o(gold, pred, punct_classes, o_class="O"):
    """
    Micro-averaged P/R/F1 over punctuation classes, class O excluded -- the
    "Overall" score of the punctuator2 tradition (Tilk & Alumäe 2016;
    Sazhok et al. 2021). O is needed only structurally, to count false
    positives / negatives of real marks.

    gold == pred == O                      -> ignored;
    gold in punct_classes, gold != pred    -> FN for the gold class;
    pred in punct_classes, gold != pred    -> FP for the predicted class.
    Both are counted independently (confusing two real marks yields one FN
    and one FP).
    """
    punct = set(punct_classes)
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        if g == p:
            if g in punct:
                tp += 1
        else:
            if g in punct:
                fn += 1
            if p in punct:
                fp += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# --------------------------------------------------------------------------
# Bootstrap CI (resampling SEGMENTS)
# --------------------------------------------------------------------------

def bootstrap_ci(segments, key_gold, key_pred, classes, metric_fn, n_boot=1000,
                 alpha=0.05, seed=42):
    """
    metric_fn(gold_flat, pred_flat) -> number.
    Resamples SEGMENTS with replacement and returns
    (point_estimate, ci_low, ci_high).
    """
    rng = random.Random(seed)
    n = len(segments)
    if n == 0:
        return 0.0, 0.0, 0.0

    point_gold = [t for s in segments for t in s[key_gold]]
    point_pred = [t for s in segments for t in s[key_pred]]
    point = metric_fn(point_gold, point_pred)

    boot_values = []
    for _ in range(n_boot):
        sample = [segments[rng.randrange(n)] for _ in range(n)]
        g_flat = [t for s in sample for t in s[key_gold]]
        p_flat = [t for s in sample for t in s[key_pred]]
        boot_values.append(metric_fn(g_flat, p_flat))

    boot_values.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return point, boot_values[lo_idx], boot_values[hi_idx]


def bootstrap_cap_precision(segments, n_boot=1000, alpha=0.05, seed=42):
    """
    Capitalization precision = TP / (TP + FP): TP -- correctly restored true
    CAP, FP -- false capitalization (see mid_sentence_cap).
    Bootstrap over segments. Returns (point, ci_low, ci_high).
    """
    counts = [(sum(1 for t in s["cap_eval_pred"] if t == "CAP"), s["cap_false"])
              for s in segments]

    def prec(cs):
        tp = sum(c[0] for c in cs)
        fp = sum(c[1] for c in cs)
        return tp / (tp + fp) if (tp + fp) else 0.0

    n = len(counts)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    vals = sorted(prec([counts[rng.randrange(n)] for _ in range(n)])
                  for _ in range(n_boot))
    lo = max(0, min(int((alpha / 2) * n_boot), n_boot - 1))
    hi = max(0, min(int((1 - alpha / 2) * n_boot) - 1, n_boot - 1))
    return prec(counts), vals[lo], vals[hi]


def accuracy_metric():
    def _fn(gold_flat, pred_flat):
        return accuracy(gold_flat, pred_flat)
    return _fn


def overall_f1_excl_o_metric(punct_classes):
    def _fn(gold_flat, pred_flat):
        _, _, f1 = micro_prf1_excl_o(gold_flat, pred_flat, punct_classes)
        return f1
    return _fn


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Punctuation & capitalization metrics with segment-level bootstrap CI.")
    ap.add_argument("--gold", type=Path, required=True, help="directory with gold .tsv")
    ap.add_argument("--pred", type=Path, required=True,
                    help="directory with prediction .tsv (same format and file names)")
    ap.add_argument("--track", choices=["A", "B"], default="A",
                    help="A: O/COMMA/PERIOD/QUESTION; B: all classes found in gold (default: A)")
    ap.add_argument("--model", default="model", help="model name, for the report and log")
    ap.add_argument("--config", default="default", help="configuration label, for the log")
    ap.add_argument("--corpus", default="unspecified", help="test set name, for the log")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=Path, default=None,
                    help="CSV file to append the result row to")
    ap.add_argument("--skip-cap", action="store_true",
                    help="do not evaluate capitalization (for models that do not restore it)")
    ap.add_argument("--cap-include-segment-start", action="store_true",
                    help="also score the first token of each segment "
                         "(excluded by default)")
    args = ap.parse_args()

    segments, missing_pred, extra_pred, mismatches = load_segments(
        args.gold, args.pred,
        exclude_segment_start=not args.cap_include_segment_start)

    if missing_pred:
        print(f"WARNING: no prediction for {len(missing_pred)} gold segment(s): "
              f"{missing_pred[:5]}{'...' if len(missing_pred) > 5 else ''}", file=sys.stderr)
    if extra_pred:
        print(f"WARNING: {len(extra_pred)} prediction file(s) without gold: "
              f"{extra_pred[:5]}{'...' if len(extra_pred) > 5 else ''}", file=sys.stderr)
    if mismatches:
        print(f"WARNING: {len(mismatches)} segment(s) excluded because the number "
              f"of tokens differs from gold:", file=sys.stderr)
        for seg_id, n_g, n_p in mismatches[:10]:
            print(f"    {seg_id}: gold={n_g} tokens, pred={n_p} tokens", file=sys.stderr)
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more", file=sys.stderr)

    if not segments:
        print("No comparable segments found. Stopping.", file=sys.stderr)
        sys.exit(1)

    if TRACK_CLASSES[args.track] is not None:
        classes = TRACK_CLASSES[args.track]
    else:
        classes = sorted({t for s in segments for t in s["gold_punct"]})

    all_gold_punct = [t for s in segments for t in s["gold_punct"]]
    all_pred_punct = [t for s in segments for t in s["pred_punct"]]

    punct_classes = [c for c in classes if c != "O"]
    per_class = prf1_per_class(all_gold_punct, all_pred_punct, punct_classes)
    overall_p, overall_r, overall_f1 = micro_prf1_excl_o(
        all_gold_punct, all_pred_punct, punct_classes)

    if not args.skip_cap:
        cap_gold = [t for s in segments for t in s["cap_eval_gold"]]
        cap_pred = [t for s in segments for t in s["cap_eval_pred"]]
        cap_total = len(cap_gold)
        cap_false = sum(s["cap_false"] for s in segments)
        cap_correct = sum(1 for g, p in zip(cap_gold, cap_pred) if g == p)
        cap_prec, cap_prec_lo, cap_prec_hi = bootstrap_cap_precision(
            segments, n_boot=args.n_boot, seed=args.seed)
        cap_pct, cap_lo, cap_hi = bootstrap_ci(
            segments, "cap_eval_gold", "cap_eval_pred", ["CAP"],
            accuracy_metric(), n_boot=args.n_boot, seed=args.seed)

    overall_point, overall_lo, overall_hi = bootstrap_ci(
        segments, "gold_punct", "pred_punct", classes,
        overall_f1_excl_o_metric(punct_classes), n_boot=args.n_boot, seed=args.seed)

    # ---- console report ----
    print(f"\n=== {args.model} | track {args.track} | config={args.config} | "
          f"corpus={args.corpus} ===")
    print(f"Segments evaluated: {len(segments)}  (excluded by length mismatch: {len(mismatches)})")
    print(f"\nPunctuation ({', '.join(punct_classes)}; class O not reported):")
    print(f"  {'class':<10} {'P':>7} {'R':>7} {'F1':>7} {'N':>7}")
    for c in punct_classes:
        p, r, f1, n = per_class[c]
        print(f"  {c:<10} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {n:>7d}")
    print("\n  Overall (micro, excl. O):")
    print(f"  {'':<10} {overall_p:>7.3f} {overall_r:>7.3f} {overall_f1:>7.3f}")
    print(f"  Overall F1 95% CI (bootstrap, {args.n_boot} resamples over segments): "
          f"[{overall_lo:.3f}, {overall_hi:.3f}]  (point={overall_point:.3f})")

    if not args.skip_cap:
        print("\nCapitalization (sentence-internal tokens only):")
        print(f"  True CAP:        {cap_total}")
        print(f"  Restored:        {cap_correct}")
        print(f"  False CAP:       {cap_false}")
        print(f"  Recall:    {cap_pct:.3f}  95% CI [{cap_lo:.3f}, {cap_hi:.3f}]")
        print(f"  Precision: {cap_prec:.3f}  95% CI [{cap_prec_lo:.3f}, {cap_prec_hi:.3f}]")
    else:
        print("\nCapitalization: skipped (--skip-cap)")

    # ---- CSV log ----
    if args.log:
        row = {
            "model": args.model,
            "track": args.track,
            "config": args.config,
            "corpus": args.corpus,
            "n_segments": len(segments),
            "n_mismatched_excluded": len(mismatches),
        }
        for c in punct_classes:
            p, r, f1, n = per_class[c]
            row[f"P_{c}"] = round(p, 4)
            row[f"R_{c}"] = round(r, 4)
            row[f"F1_{c}"] = round(f1, 4)
            row[f"N_{c}"] = n
        row["overall_P_excl_O"] = round(overall_p, 4)
        row["overall_R_excl_O"] = round(overall_r, 4)
        row["overall_F1_excl_O"] = round(overall_f1, 4)
        row["overall_F1_excl_O_ci_lo"] = round(overall_lo, 4)
        row["overall_F1_excl_O_ci_hi"] = round(overall_hi, 4)
        if not args.skip_cap:
            row["cap_total"] = cap_total
            row["cap_correct"] = cap_correct
            row["cap_false"] = cap_false
            row["cap_correct_rate"] = round(cap_pct, 4)
            row["cap_precision"] = round(cap_prec, 4)
            row["cap_precision_ci_lo"] = round(cap_prec_lo, 4)
            row["cap_precision_ci_hi"] = round(cap_prec_hi, 4)
            row["cap_correct_rate_ci_lo"] = round(cap_lo, 4)
            row["cap_correct_rate_ci_hi"] = round(cap_hi, 4)
        else:
            for k in ("cap_total", "cap_correct", "cap_false", "cap_correct_rate",
                      "cap_precision", "cap_precision_ci_lo", "cap_precision_ci_hi",
                      "cap_correct_rate_ci_lo", "cap_correct_rate_ci_hi"):
                row[k] = "n/a"

        file_exists = args.log.exists()
        if file_exists:
            with args.log.open(encoding="utf-8") as f:
                existing_fields = next(csv.reader(f))
            fieldnames = list(dict.fromkeys(existing_fields + list(row.keys())))
        else:
            fieldnames = list(row.keys())

        rows = []
        if file_exists:
            with args.log.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        rows.append(row)
        with args.log.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nResult appended to {args.log.resolve()}")


if __name__ == "__main__":
    main()
