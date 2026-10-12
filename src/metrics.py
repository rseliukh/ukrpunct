#!/usr/bin/env python3
"""
metrics.py
==========
Єдиний скрипт оцінки для всіх п'яти моделей (punctuator2, BERT ukr, BERT-47,
qwen3:1.7b, qwen3:4b) і обох треків (A/B) — пишеться один раз, не переписується
під кожну модель (розділ 5, тиждень 2 плану).

ВХІД: gold-каталог і pred-каталог з ОДНАКОВИМИ іменами файлів (segment_id.tsv),
у форматі, який видає build_corpus.py:
    token\tpunct_tag\tcap_tag
pred-файли повинні бути ВИРІВНЯНІ з gold token-в-token (однакова кількість
рядків, той самий порядок слів) — вирівнювання виходу конкретної моделі до
цього формату виконує обгортка predict() цієї моделі (розділ 8.1 плану:
predict(text) -> tags), а не цей скрипт. metrics.py лише рахує метрики.

Рахує:
  - Precision/Recall/F1 по кожному класу пунктуації, БЕЗ класу O (в межах треку)
  - Overall P/R/F1 (micro, excl. O) — ГОЛОВНА метрика; саме так визначає
    "Overall" punctuator2-лінія робіт (Tilk & Alumäe 2016; Sazhok et al. 2021
    "Punctuation Restoration for Ukrainian Broadcast Speech...") — клас O не
    звітується окремо і не входить у підсумкову метрику ніде в цій традиції
  - Капіталізація: лише СЕРЕДИНА РЕЧЕННЯ. Береться кількість gold-токенів з CAP,
    які НЕ стоять одразу після PERIOD/QUESTION (gold-пунктуація попереднього
    токена) — тобто справжня, змістовна капіталізація (власні назви,
    абревіатури тощо); і скільки з них модель відновила правильно.
    Звітується: загальна кількість, кількість правильних, відсоток, а також
    кількість ХИБНОЇ капіталізації (gold=O, pred=CAP) на тих самих позиціях.
  - Bootstrap CI (за замовчуванням 1000 ресемплів, на рівні СЕГМЕНТІВ,
    не окремих токенів — щоб не порушувати незалежність спостережень) —
    порахований для Overall(excl.O) і для відсотка правильної капіталізації
  - (опційно) дописує рядок результату в спільний results_log.csv —
    структуроване логування прогонів (розділ 8.1: "логувати кожен прогін")

Використання:
    python3 metrics.py --gold out/golden_track_a --pred preds/punctuator2_track_a \
        --track A --model punctuator2 --config "baseline" --corpus bruk_fiction \
        --log results_log.csv
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

TRACK_CLASSES = {
    "A": ["O", "COMMA", "PERIOD", "QUESTION"],
    "B": None,  # визначається динамічно з даних (усі класи, що трапляються)
}

# Теги, після яких капіталізація слова тривіальна (початок нового речення)
SENT_END_TAGS = {"PERIOD", "QUESTION"}


# --------------------------------------------------------------------------
# Завантаження TSV
# --------------------------------------------------------------------------

def read_tsv(path: Path):
    """Повертає (tokens, punct_tags, cap_tags) списками, у порядку рядків."""
    tokens, punct, cap = [], [], []
    with path.open(encoding="utf-8") as f:
        header = f.readline()  # token\tpunct_tag\tcap_tag
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}: очікував 3 колонки, отримав {parts!r}")
            tok, p, c = parts
            tokens.append(tok)
            punct.append(p)
            cap.append(c)
    return tokens, punct, cap


def load_segments(gold_dir: Path, pred_dir: Path, exclude_segment_start=True):
    """
    Повертає список сегментів: кожен -- dict(segment_id, gold_punct, pred_punct,
    gold_cap, pred_cap, cap_eval_gold, cap_eval_pred, cap_false). Сегменти з невідповідною довжиною (модель "з'їла" чи
    додала слів) НЕ відкидаються мовчки -- звітуються окремо, щоб було видно
    масштаб проблеми вирівнювання для конкретної моделі.
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
    Оцінка капіталізації лише всередині речення. Позиції, де попередній
    gold-токен має PERIOD/QUESTION (початок речення), виключаються повністю --
    і для справжніх, і для хибних випадків (gold, а не pred, щоб набір позицій
    був однаковий для всіх моделей).
    Повертає (gold_eval, pred_eval, n_false):
      gold_eval/pred_eval -- паралельні списки для gold-токенів з CAP
        (gold_eval суто "CAP", pred_eval -- що передбачила модель);
      n_false -- хибна капіталізація: gold=O, pred=CAP на тих самих позиціях,
        КРІМ випадків, коли попередній токен у ПЕРЕДБАЧЕННІ моделі має
        PERIOD/QUESTION: велика літера там -- наслідок хибної межі речення
        (помилка пунктуації), а не окрема помилка відновлення регістру.
    Перший токен сегмента за замовчуванням ВИКЛЮЧАЄТЬСЯ (початок фрагмента --
    це початок речення, велика літера там тривіальна); exclude_segment_start=False
    залишає його в підрахунку.
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
    gold, pred: паралельні списки тегів (об'єднані по всіх сегментах).
    Повертає dict: class -> (precision, recall, f1, support).
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
    Micro-averaged P/R/F1 ПО КЛАСАХ ПУНКТУАЦІЇ, БЕЗ класу O -- це і є "Overall"
    у прийнятому в літературі сенсі (Tilk & Alumäe 2016; punctuator2;
    Sazhok et al. 2021 "Punctuation Restoration for Ukrainian Broadcast
    Speech..."): жодна з цих робіт не звітує P/R/F1 для O окремо і не включає
    його в підсумкову метрику -- клас потрібен лише структурно (щоб коректно
    рахувати false positives/negatives для реальних знаків пунктуації), сам
    по собі він не звітується, бо майже завжди близький до 1.0 (домінує за
    кількістю токенів) і робить будь-яку метрику з ним включеним оманливо
    завищеною відносно цих орієнтирів.

    gold==pred==O -- ігнорується (не FP, не FN: правильне "нічого").
    gold in punct_classes, gold!=pred -- FN для gold-класу.
    pred in punct_classes, gold!=pred -- FP для pred-класу.
    Обидва рахуються незалежно (той самий рядок може дати і FN, і FP,
    якщо переплутано два реальних знаки пунктуації між собою).
    """
    punct = set(punct_classes)
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        if g == p:
            if g in punct:
                tp += 1
            # g==p==O -- пропускаємо, не рахується
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
# Bootstrap CI (ресемпл на рівні СЕГМЕНТІВ)
# --------------------------------------------------------------------------

def bootstrap_ci(segments, key_gold, key_pred, classes, metric_fn, n_boot=1000,
                  alpha=0.05, seed=42):
    """
    metric_fn(gold_flat, pred_flat) -> число (напр. Overall F1 або частка правильних).
    Ресемплюємо СЕГМЕНТИ з поверненням (не окремі токени), рахуємо метрику
    на кожному ресемплі, повертаємо (point_estimate, ci_low, ci_high).
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
    Precision капіталізації = TP / (TP + FP), де TP -- правильно відновлені
    справжні CAP, FP -- хибна капіталізація (див. mid_sentence_cap).
    Bootstrap на рівні сегментів. Повертає (point, ci_low, ci_high).
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True, help="каталог golden .tsv")
    ap.add_argument("--pred", type=Path, required=True, help="каталог pred .tsv (той самий формат)")
    ap.add_argument("--track", choices=["A", "B"], required=True)
    ap.add_argument("--model", required=True, help="назва моделі, для логу")
    ap.add_argument("--config", default="default", help="конфіг (напр. temp=0,zero-shot)")
    ap.add_argument("--corpus", default="unspecified", help="назва корпусу/test set")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=Path, default=None,
                     help="CSV-файл для дописування результату (спільний лог усіх прогонів)")
    ap.add_argument("--skip-cap", action="store_true",
                     help="не оцінювати капіталізацію (для моделей типу punctuator2, "
                          "які її не відновлюють -- інакше результат буде нечесно занижений)")
    ap.add_argument("--cap-include-segment-start", action="store_true",
                     help="залишити в оцінці капіталізації і перший токен кожного "
                          "сегмента (за замовчуванням він виключається)")
    args = ap.parse_args()

    segments, missing_pred, extra_pred, mismatches = load_segments(
        args.gold, args.pred,
        exclude_segment_start=not args.cap_include_segment_start)

    if missing_pred:
        print(f"⚠ Немає pred-файлу для {len(missing_pred)} gold-сегмент(ів): "
              f"{missing_pred[:5]}{'...' if len(missing_pred) > 5 else ''}", file=sys.stderr)
    if extra_pred:
        print(f"⚠ {len(extra_pred)} pred-файл(ів) без відповідного gold: "
              f"{extra_pred[:5]}{'...' if len(extra_pred) > 5 else ''}", file=sys.stderr)
    if mismatches:
        print(f"⚠ {len(mismatches)} сегмент(ів) виключено через невідповідність "
              f"довжини (модель додала/загубила токени):", file=sys.stderr)
        for seg_id, n_g, n_p in mismatches[:10]:
            print(f"    {seg_id}: gold={n_g} токенів, pred={n_p} токенів", file=sys.stderr)
        if len(mismatches) > 10:
            print(f"    ... і ще {len(mismatches) - 10}", file=sys.stderr)

    if not segments:
        print("Жодного сумісного сегмента для оцінки не знайдено. Зупиняюсь.", file=sys.stderr)
        sys.exit(1)

    # класи пунктуації для цього треку
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
        # gold_eval складається лише з CAP, тож accuracy = частка відновлених
        cap_prec, cap_prec_lo, cap_prec_hi = bootstrap_cap_precision(
            segments, n_boot=args.n_boot, seed=args.seed)
        cap_pct, cap_lo, cap_hi = bootstrap_ci(
            segments, "cap_eval_gold", "cap_eval_pred", ["CAP"],
            accuracy_metric(), n_boot=args.n_boot, seed=args.seed)

    # bootstrap CI на Overall F1 (excl. O -- ГОЛОВНА метрика, порівнянна з
    # літературою) і на відсоток правильної капіталізації (вище)
    overall_point, overall_lo, overall_hi = bootstrap_ci(
        segments, "gold_punct", "pred_punct", classes,
        overall_f1_excl_o_metric(punct_classes), n_boot=args.n_boot, seed=args.seed)

    # ---- звіт у консоль ----
    print(f"\n=== {args.model} | track {args.track} | config={args.config} | "
          f"corpus={args.corpus} ===")
    print(f"Сегментів оцінено: {len(segments)}  (пропущено через mismatch: {len(mismatches)})")
    print(f"\nПунктуація ({', '.join(punct_classes)}; клас O не звітується окремо -- "
          f"стандартна практика, див. docstring micro_prf1_excl_o):")
    print(f"  {'клас':<10} {'P':>7} {'R':>7} {'F1':>7} {'N':>7}")
    for c in punct_classes:
        p, r, f1, n = per_class[c]
        print(f"  {c:<10} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {n:>7d}")
    print(f"\n  Overall (excl. O, micro; ГОЛОВНА метрика для порівняння з "
          f"літературою -- Tilk&Alumäe/punctuator2/Sazhok et al.):")
    print(f"  {'':<10} {overall_p:>7.3f} {overall_r:>7.3f} {overall_f1:>7.3f}")
    print(f"  Overall-F1 bootstrap {args.n_boot} ресемплів, 95% CI: "
          f"[{overall_lo:.3f}, {overall_hi:.3f}]  (point={overall_point:.3f})")

    if not args.skip_cap:
        print(f"\nКапіталізація (лише всередині речення; слова після PERIOD/QUESTION виключено):")
        print(f"  Справжніх випадків капіталізації: {cap_total}")
        print(f"  Відновлено правильно:             {cap_correct}  "
              f"({cap_pct:.3f})")
        print(f"  Хибна капіталізація (gold=O, pred=CAP, без слів після "
              f"передбаченого PERIOD/QUESTION): {cap_false}")
        print(f"  Recall (відновлено / справжніх): {cap_pct:.3f}  "
              f"95% CI [{cap_lo:.3f}, {cap_hi:.3f}]")
        print(f"  Precision (відновлено / (відновлено + хибних)): {cap_prec:.3f}  "
              f"95% CI [{cap_prec_lo:.3f}, {cap_prec_hi:.3f}]")
        print(f"  (bootstrap {args.n_boot} ресемплів по сегментах)")
    else:
        print("\nКапіталізація: пропущено (--skip-cap)")

    # ---- структурований лог (розділ 8.1 плану) ----
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
        # ГОЛОВНА метрика (порівнянна з Tilk&Alumäe/punctuator2/Sazhok et al.)
        row["overall_P_excl_O"] = round(overall_p, 4)
        row["overall_R_excl_O"] = round(overall_r, 4)
        row["overall_F1_excl_O"] = round(overall_f1, 4)
        row["overall_F1_excl_O_ci_lo"] = round(overall_lo, 4)
        row["overall_F1_excl_O_ci_hi"] = round(overall_hi, 4)
        # капіталізація всередині речення (без слів після PERIOD/QUESTION)
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
        # якщо лог уже існує з іншим набором класів (інший трек), об'єднуємо колонки
        if file_exists:
            with args.log.open(encoding="utf-8") as f:
                existing_fields = next(csv.reader(f))
            fieldnames = list(dict.fromkeys(existing_fields + list(row.keys())))
        else:
            fieldnames = list(row.keys())

        # переписуємо файл цілком, якщо з'явились нові колонки (rare, безпечно для невеликого лога)
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
        print(f"\n→ Результат дописано в {args.log.resolve()}")


if __name__ == "__main__":
    main()
