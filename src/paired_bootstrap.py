#!/usr/bin/env python3
"""
paired_bootstrap.py
===================
Парний bootstrap різниці Overall micro F1 (excl. O) між моделями, Track A.

На кожній ітерації вибирається ОДИН набір індексів сегментів (з повторенням),
спільний для обох моделей; F1 рахується з сумарних TP/FP/FN по ресемплу
(так само, як micro_prf1_excl_o у metrics.py).

Читання даних — через metrics.load_segments(), тобто ті самі gold/pred
каталоги, той самий формат і ті самі виключення, що й у таблицях результатів.

Перед порівняннями скрипт перевіряє, що точкові F1 відтворюють таблиці
exp1_results.md / exp2_results.md (EXPECTED нижче).

Використання (з каталогу, де лежить metrics.py, або з --metrics-dir):
    python3 paired_bootstrap.py --exp 1 \
        --gold-template "out/golden_track_a/{corpus}" \
        --pred-template "results/predictions/{model}/{corpus}"

    python3 paired_bootstrap.py --exp 2 --gold-template ... --pred-template ...

Шаблони підставляють {model} і {corpus}; підлаштуйте під свою структуру.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PUNCT_CLASSES = ("COMMA", "PERIOD", "QUESTION")

EXPERIMENTS = {
    "1": {
        "corpora": ["bruk_fiction", "bruk_other"],
        "pairs": [
            ("mamay27b", "gemma4_26b"),
            ("gemma4_26b", "qwen14b"),
            ("gemma4_26b", "lapa"),
            ("qwen14b", "lapa"),
            ("lapa", "bert_47"),
            ("qwen14b", "bert_47"),
            ("mamay27b", "bert_47"),
            ("gemma4_26b", "bert_47"),
            ("bert_47", "bert_ukr"),
            ("bert_ukr", "pnct2"),
        ],
    },
    "2": {
        "corpora": ["bruk_fiction_oracle", "bruk_other_oracle"],
        "pairs": [
            ("bert_uk2", "gemma4_26b"),
            ("bert_uk2", "mamay27b"),
            ("gemma4_26b", "mamay27b"),
            ("mamay27b", "bert_47"),
            ("bert_uk2", "bert_47"),
            ("gemma4_26b", "bert_47"),
            ("gemma4_26b", "lapa"),
            ("bert_47", "lapa"),
            ("lapa", "qwen14b"),
            ("qwen14b", "bert_ukr"),
            ("bert_ukr", "pnct2"),
        ],
    },
}

# Точкові Overall F1 з таблиць результатів — для перевірки відтворення.
EXPECTED = {
    "bruk_fiction": {"mamay27b": 0.827, "gemma4_26b": 0.808, "qwen14b": 0.795,
                     "lapa": 0.779, "bert_47": 0.751, "bert_ukr": 0.650, "pnct2": 0.627},
    "bruk_other": {"mamay27b": 0.885, "gemma4_26b": 0.858, "qwen14b": 0.855,
                   "lapa": 0.858, "bert_47": 0.817, "bert_ukr": 0.726, "pnct2": 0.706},
    "bruk_fiction_oracle": {"bert_uk2": 0.930, "gemma4_26b": 0.906, "mamay27b": 0.890,
                            "bert_47": 0.882,
                            "lapa": 0.848, "qwen14b": 0.813, "bert_ukr": 0.770,
                            "pnct2": 0.492},
    "bruk_other_oracle": {"bert_uk2": 0.927, "gemma4_26b": 0.931, "mamay27b": 0.944,
                          "bert_47": 0.902,
                          "lapa": 0.911, "qwen14b": 0.878, "bert_ukr": 0.797,
                          "pnct2": 0.516},
}


def segment_counts(gold, pred, classes=PUNCT_CLASSES, skip=()):
    """TP/FP/FN одного сегмента — та сама логіка, що micro_prf1_excl_o.
    skip — позиції токенів, які не враховуються (маска невирівняних)."""
    punct = set(classes)
    tp = fp = fn = 0
    for i, (g, p) in enumerate(zip(gold, pred)):
        if i in skip:
            continue
        if g == p:
            if g in punct:
                tp += 1
        else:
            if g in punct:
                fn += 1
            if p in punct:
                fp += 1
    return tp, fp, fn


def load_segment_counts(metrics, gold_dir, pred_dir, classes=PUNCT_CLASSES, mask=None):
    """Повертає (segment_ids, np.array (n, 3) з TP, FP, FN) у відсортованому
    порядку сегментів. Зупиняється, якщо є пропущені/зайві файли чи mismatch
    довжини — інакше порушиться equal-N."""
    for d in (gold_dir, pred_dir):
        if not d.is_dir():
            sys.exit(f"Немає каталогу: {d}")
    segments, missing, extra, mismatches = metrics.load_segments(gold_dir, pred_dir)
    if missing or mismatches:
        sys.exit(f"{pred_dir}: equal-N порушено — missing={len(missing)}, "
                 f"length mismatch={len(mismatches)}")
    if extra:
        # Як і metrics.py: файли без gold-пари ігноруються (напр., залишки
        # прогону на старішій версії набору); оцінка — лише на gold-сегментах.
        print(f"  ⚠ {pred_dir}: {len(extra)} pred-файл(ів) без gold-пари "
              f"ігноровано, напр. {extra[:3]}", file=sys.stderr)
    ids = [s["segment_id"] for s in segments]
    mask = mask or {}
    counts = np.array([segment_counts(s["gold_punct"], s["pred_punct"], classes,
                                      mask.get(s["segment_id"], ()))
                       for s in segments], dtype=np.int64)
    return ids, counts


def read_unaligned(path, gold_dir, metrics):
    """Читає <model>/<corpus>.unaligned.tsv -> dict seg_id -> set(idx, 0-based).
    База idx (0 чи 1) визначається звіркою колонки token з gold."""
    rows = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0]:
                continue
            rows.append((parts[col["seg_id"]], int(parts[col["idx"]]),
                         parts[col["token"]].lower() if "token" in col else None,
                         parts[col["opcode"]] if "opcode" in col else ""))
    gold_cache = {}

    def gold_tokens(seg):
        if seg not in gold_cache:
            gp = gold_dir / f"{seg}.tsv"
            gold_cache[seg] = [t.lower() for t in metrics.read_tsv(gp)[0]] if gp.exists() else None
        return gold_cache[seg]

    hits = {0: 0, 1: 0}
    checked = 0
    for seg, idx, tok, _ in rows:
        g = gold_tokens(seg)
        if g is None or tok is None:
            continue
        checked += 1
        for base in (0, 1):
            j = idx - base
            if 0 <= j < len(g) and g[j] == tok:
                hits[base] += 1
    base = max(hits, key=hits.get)
    rate = hits[base] / checked if checked else 1.0
    if checked and rate < 0.9:
        sys.exit(f"{path}: колонка token збігається з gold лише в {rate:.0%} рядків "
                 f"(база 0: {hits[0]}, база 1: {hits[1]} з {checked}) — перевірте формат")
    out, skipped = {}, 0
    for seg, idx, _, _ in rows:
        if gold_tokens(seg) is None:
            skipped += 1  # сегмент не з цього gold-набору (напр. стара версія)
            continue
        out.setdefault(seg, set()).add(idx - base)
    n = sum(len(v) for v in out.values())
    return out, {"base": base, "match": rate, "n": n, "skipped_rows": skipped}


def resolve_pred_dir(pred_dir):
    """Якщо в каталозі немає .tsv, але є converted/ з .tsv (pnct2) — беремо його."""
    if any(pred_dir.glob("*.tsv")):
        return pred_dir
    conv = pred_dir / "converted"
    if conv.is_dir() and any(conv.glob("*.tsv")):
        return conv
    return pred_dir


def f1(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def paired_bootstrap(ca, cb, idx):
    """idx — масив (B, n) індексів ресемплу, спільний для всіх пар корпусу."""
    delta = f1(ca[idx].sum(axis=1)) - f1(cb[idx].sum(axis=1))
    point = float(f1(ca.sum(0)) - f1(cb.sum(0)))
    lo, hi = np.percentile(delta, [2.5, 97.5])
    p = 2 * min((delta <= 0).mean(), (delta >= 0).mean())
    return {"delta": point, "ci": (float(lo), float(hi)), "p": float(min(p, 1.0))}


def holm(pvals):
    """Поправка Холма. pvals: dict name -> p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=sorted(EXPERIMENTS), required=True)
    ap.add_argument("--gold-template", required=True,
                    help='шлях до gold-каталогу з {corpus}, напр. "out/golden_track_a/{corpus}"')
    ap.add_argument("--pred-template", required=True,
                    help='шлях до pred-каталогу з {model} і {corpus}')
    ap.add_argument("--pred-override", action="append", default=[],
                    metavar="MODEL=TEMPLATE",
                    help="окремий шаблон pred-каталогу для моделі; можна повторювати")
    ap.add_argument("--metrics-dir", type=Path, default=Path(__file__).resolve().parent,
                    help="каталог, де лежить metrics.py")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask-unaligned", action="store_true",
                    help="виключити позиції, невирівняні хоч в одній LLM, для ВСІХ моделей "
                         "(файли <pred-каталог>.unaligned.tsv)")
    ap.add_argument("--tol", type=float, default=0.0005,
                    help="допуск для перевірки точкових F1 проти таблиць")
    args = ap.parse_args()

    sys.path.insert(0, str(args.metrics_dir))
    import metrics  # noqa: E402

    overrides = {}
    for item in args.pred_override:
        model, _, tmpl = item.partition("=")
        if not tmpl:
            sys.exit(f"--pred-override очікує MODEL=TEMPLATE, отримав {item!r}")
        overrides[model] = tmpl

    cfg = EXPERIMENTS[args.exp]
    ok = True
    for corpus in cfg["corpora"]:
        gold_dir = Path(args.gold_template.format(corpus=corpus))
        models = list(dict.fromkeys(m for pair in cfg["pairs"] for m in pair))

        mask = None
        if args.mask_unaligned:
            mask = {}
            print(f"\n[{corpus}] маска невирівняних токенів:")
            for m in models:
                tmpl = overrides.get(m, args.pred_template)
                up = Path(str(Path(tmpl.format(model=m, corpus=corpus))) + ".unaligned.tsv")
                if not up.exists():
                    continue
                um, info = read_unaligned(up, gold_dir, metrics)
                for seg, idxs in um.items():
                    mask.setdefault(seg, set()).update(idxs)
                print(f"  {m:12s} {info['n']:5d} позицій  (idx з {info['base']}, "
                      f"збіг token з gold {info['match']:.1%}"
                      f"{', поза gold-набором рядків: ' + str(info['skipped_rows']) if info['skipped_rows'] else ''})")
            n_mask = sum(len(v) for v in mask.values())
            n_tok = n_punct = 0
            for gp in gold_dir.glob("*.tsv"):
                _, gpunct, _ = metrics.read_tsv(gp)
                n_tok += len(gpunct)
                n_punct += sum(1 for i in mask.get(gp.stem, ()) if i < len(gpunct)
                               and gpunct[i] in PUNCT_CLASSES)
            print(f"  об'єднана маска: {n_mask} з {n_tok} токенів ({n_mask / n_tok:.2%}); "
                  f"з них із gold-пунктуацією: {n_punct}")

        data, unmasked = {}, {}
        for m in models:
            tmpl = overrides.get(m, args.pred_template)
            pred_dir = Path(tmpl.format(model=m, corpus=corpus))
            if not pred_dir.is_dir():
                print(f"[{corpus}] ⚠ пропускаю {m}: немає {pred_dir}", file=sys.stderr)
                continue
            pred_dir = resolve_pred_dir(pred_dir)
            print(f"[{corpus}] {m}: {pred_dir}")
            data[m] = load_segment_counts(metrics, gold_dir, pred_dir, mask=mask)
            if mask is not None:
                unmasked[m] = float(f1(load_segment_counts(metrics, gold_dir, pred_dir)[1].sum(0)))

        # 1) Перевірка відтворення таблиць
        title = ("точкові Overall F1: усі токени → без невирівняних"
                 if mask is not None else "точкові Overall F1 (перевірка проти таблиць)")
        print(f"\n== {corpus} == {title}")
        for m, (_, c) in data.items():
            got = float(f1(c.sum(0)))
            exp = EXPECTED.get(corpus, {}).get(m)
            flag = ""
            if mask is not None:
                u = unmasked[m]
                print(f"  {m:12s} {u:.4f} → {got:.4f}  (Δ {got - u:+.4f})")
                continue
            elif exp is not None and abs(round(got, 3) - exp) > args.tol:
                flag, ok = f"  ✗ у таблиці {exp:.3f}", False
            print(f"  {m:12s} {got:.4f}{flag}")

        # 2) Парні порівняння на спільному ресемплі
        ids0 = next(iter(data.values()))[0]
        for m, (ids, _) in data.items():
            if ids != ids0:
                sys.exit(f"[{corpus}] набір/порядок сегментів {m} відрізняється")
        n = len(ids0)
        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, n, size=(args.n_boot, n))

        res = {}
        for a, b in cfg["pairs"]:
            if a in data and b in data:
                res[f"{a} − {b}"] = paired_bootstrap(data[a][1], data[b][1], idx)
        adj = holm({k: v["p"] for k, v in res.items()})

        print(f"\n== {corpus} == парний bootstrap, N={n} сегментів, "
              f"{args.n_boot} ресемплів, поправка Холма")
        print(f"  {'пара':28s} {'Δ F1':>7s}  {'95% CI':>18s}  {'p':>7s}  {'p_Holm':>7s}")
        for k, r in res.items():
            sig = "*" if adj[k] < 0.05 else ""
            print(f"  {k:28s} {r['delta']:+.3f}  "
                  f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  "
                  f"{r['p']:7.4f}  {adj[k]:7.4f} {sig}")

    if not ok:
        print("\n✗ Деякі точкові F1 не збіглися з таблицями — перевірте шляхи/версії "
              "прогонів, перш ніж використовувати Δ.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
