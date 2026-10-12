# UkrPunct

Test sets and evaluation scripts for automatic punctuation and capitalization
restoration in Ukrainian text.

| Subset | Source | Track | Status |
|---|---|---|---|
| **UkrPunct-BRUK** | Brown Corpus of Ukrainian (BRUK) | A | v1.0 (this release) |
| UkrPunct-Rada | Verkhovna Rada transcripts | B | planned |

UkrPunct-BRUK was introduced in:

> R. Seliukh, M. Sazhok, O. Yukhymenko. *Punctuation and Capitalization
> Restoration for Ukrainian: Specialized Models vs. Large Language Models.*
> Інформаційні технології і автоматизація – 2026 (ITiA), ONTU, Odesa, 2026.
> TODO: pages.

The benchmark is meant for comparing models under identical conditions: the same
segments, the same tag set and the same metric code for every system.

## UkrPunct-BRUK

| Corpus | Set | Segmentation | Segments | Tokens | COMMA | PERIOD | QUESTION | Track |
|---|---|---|---:|---:|---:|---:|---:|---|
| BRUK | `fiction-window` | fixed window ≤200 words | 235 | 29 261 | 3 573 | 2 415 | 140 | A |
| BRUK | `other-window` | fixed window ≤200 words | 263 | 37 113 | 3 829 | 2 379 | 87 | A |
| BRUK | `fiction-sentence` | one sentence (oracle boundaries) | 226 | 2 226 | 252 | 113 | 113 | A |
| BRUK | `other-sentence` | one sentence (oracle boundaries) | 168 | 2 008 | 190 | 84 | 84 | A |

- **Window sets** consist of consecutive whole sentences, up to 200 words per
  segment: every window starts at the beginning of a sentence and ends with a
  sentence-final mark, but the sentence boundaries inside it are unknown to the
  model. The class distribution is natural.
- **Sentence sets** are drawn from the same chunks as the window sets (every
  sentence lies inside a window). QUESTION is balanced against PERIOD, so overall
  scores on these sets are not directly comparable with scores on natural text.
- **Fiction** is the BRUK fiction subcategory; **other** covers the remaining
  BRUK genres.

### Tracks

- **Track A** — four classes: `O` (no mark), `COMMA`, `PERIOD`, `QUESTION`,
  plus capitalization. Only sentences whose punctuation is limited to commas,
  periods, ellipses and question marks were included; ellipses were replaced
  with periods. The original BRUK punctuation and capitalization serve as the
  reference.
- **Track B** — extended punctuation set; planned for UkrPunct-Rada.

## Layout

```
data/
  bruk/
    LICENSE                 CC BY-NC-SA 4.0
    <set>/
      input/<id>.txt        model input: one line, lowercase, no punctuation
      gold/<id>.tsv         reference annotation, one token per line
      manifest.csv          source BRUK document for each segment
  rada/                     UkrPunct-Rada (planned)
src/
  metrics.py                P/R/F1 and capitalization scores with bootstrap CI
```

## Data format

**Input** (`input/<id>.txt`) — one line: the segment's words, lowercased,
punctuation removed, separated by single spaces. The apostrophe inside words is
kept (`подвір'я`). Splitting the line on whitespace gives exactly the tokens of
the gold file, in the same order.

```
намагаючись триматися темнішої частини вулиці анна тихенько прочинила хвіртку …
```

**Gold** (`gold/<id>.tsv`) — tab-separated, UTF-8, with a header row; one row
per token:

```
token	punct_tag	cap_tag
Намагаючись	O	CAP
триматися	O	O
…
вулиці	COMMA	O
Анна	O	CAP
…
досередини	PERIOD	O
```

- `punct_tag` — the mark that follows the token: `O` (none), `COMMA`, `PERIOD`,
  `QUESTION`.
- `cap_tag` — `CAP` if the word starts with a capital letter, otherwise `O`.

**Manifest** (`manifest.csv`) — one row per segment:

| Column | Meaning |
|---|---|
| `segment_id` | file name without extension |
| `chunk_id` | source chunk; segments of one chunk are consecutive text |
| `category` | BRUK genre category |
| `source` | BRUK source file |
| `mode` | segmentation: window or sentence |
| `n_segments_in_chunk` | number of segments cut from the chunk |
| `n_words` | segment length in words |
| `over_200_words` | whether the segment exceeds the 200-word window |
| `end_class` | punctuation tag of the last token |

## Evaluating your model

1. Run your model on every file in `data/bruk/<set>/input/`.
2. Write predictions in the gold format (same header and columns), one file per
   segment with the same `<id>`, into a directory of your choice:
   - the `token` column must reproduce the input tokens one-to-one. A model
     that rewrites words (typical for LLMs) must be aligned back to the input
     first. In the baselines, alignment is exact word matching after
     normalization (lowercasing, apostrophe variants, `ґ`→`г`; TODO: check against the runner); input words without a
     match get `O` and no `CAP`. Segments with a different number of tokens
     are excluded by the scorer and reported;
   - marks outside the four classes must be mapped to them. The baselines use
     this deterministic mapping: TODO (e.g. `!` → ?, `;` → ?, `:` → ?, `…` → `PERIOD`).
3. Score (Python ≥ 3.8, standard library only):

```bash
python src/metrics.py --gold data/bruk/fiction-window/gold --pred my_preds/fiction-window --model my-model
```

Useful options: `--skip-cap` for models that do not restore case,
`--log results.csv` to append a row with all scores to a CSV file,
`--n-boot` / `--seed` for the bootstrap.

### Metrics

- **Punctuation:** micro precision / recall / F1 over `COMMA`, `PERIOD`,
  `QUESTION` (class `O` excluded), plus per-class scores.
- **Capitalization:** recall and precision on sentence-internal tokens only;
  tokens following `PERIOD` or `QUESTION` and the first token of a segment are
  excluded, since capitalizing them is trivial once the boundary is known.
- **Confidence intervals:** 95%, bootstrap with 1 000 resamples over segments
  (not tokens), to respect within-segment dependence.
- **Significance** (used for the baseline comparison in the paper, not part of
  `metrics.py`): paired bootstrap over segments (10 000 resamples, the same
  resampled segments for both systems), Holm correction across the pairs
  compared within a set.

## Baseline results

Punctuation: micro F1 excluding `O`, with 95% CI.

| Model | fiction-window | other-window | fiction-sentence | other-sentence |
|---|---|---|---|---|
| *Large language models* | | | | |
| MamayLM 27B | **0.827** [0.814, 0.839] | **0.885** [0.875, 0.894] | 0.890 [0.859, 0.918] | **0.944** [0.920, 0.964] |
| Gemma 4 26B-A4B | 0.808 [0.796, 0.821] | 0.858 [0.847, 0.868] | 0.906 [0.880, 0.929] | 0.931 [0.905, 0.953] |
| Qwen3-14B | 0.795 [0.779, 0.808] | 0.855 [0.844, 0.865] | 0.813 [0.775, 0.847] | 0.878 [0.849, 0.906] |
| Lapa | 0.779 [0.757, 0.798] | 0.858 [0.845, 0.869] | 0.848 [0.813, 0.880] | 0.911 [0.881, 0.938] |
| *Specialized models* | | | | |
| xlm-roberta | 0.751 [0.736, 0.768] | 0.817 [0.806, 0.828] | 0.882 [0.851, 0.912] | 0.902 [0.860, 0.939] |
| uk-punctcase | 0.650 [0.633, 0.670] | 0.726 [0.711, 0.742] | 0.770 [0.734, 0.804] | 0.797 [0.754, 0.834] |
| punctuation_uk_bert | — ¹ | — ¹ | **0.930** [0.909, 0.948] | 0.927 [0.884, 0.964] |
| punctuator2 | 0.627 [0.611, 0.643] | 0.706 [0.692, 0.721] | 0.492 [0.453, 0.530] | 0.516 [0.468, 0.562] |

¹ Trained on single sentences; on fixed windows it places every boundary on the
last token, so it is evaluated on the sentence sets only.

Window and sentence scores are not directly comparable: the sentence sets are
balanced and contain different segments.

Mid-sentence capitalization on the window sets (recall / precision / F1):

| Model | fiction-window (767 capitals) | other-window (1580 capitals) |
|---|---|---|
| MamayLM 27B | 0.919 / 0.974 / 0.946 | 0.861 / 0.977 / 0.915 |
| Gemma 4 26B-A4B | 0.926 / 0.961 / 0.943 | 0.864 / 0.953 / 0.906 |
| Lapa | 0.876 / 0.971 / 0.921 | 0.854 / 0.959 / 0.903 |
| xlm-roberta | 0.905 / 0.899 / 0.902 | 0.858 / 0.936 / 0.895 |
| Qwen3-14B | 0.838 / 0.957 / 0.894 | 0.812 / 0.957 / 0.879 |
| uk-punctcase | 0.884 / 0.824 / 0.853 | 0.870 / 0.885 / 0.877 |

punctuator2 does not restore case.

### Models

| Model | Checkpoint | Setup |
|---|---|---|
| MamayLM 27B | [`INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-GGUF`](https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-GGUF), Q4_K_M | LLM ² |
| Gemma 4 26B-A4B | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF), UD-IQ4_XS | LLM ², thinking off |
| Qwen3-14B | `qwen3:14b` (Ollama library) | LLM ², TODO: thinking mode |
| Lapa | [`lapa-llm/lapa-v0.1.2-instruct-GGUF`](https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct-GGUF), Q4_K_M | LLM ² |
| xlm-roberta | [`1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase) | token classification |
| uk-punctcase | [`ukr-models/uk-punctcase`](https://huggingface.co/ukr-models/uk-punctcase) | token classification |
| punctuation_uk_bert | [`dchaplinsky/punctuation_uk_bert`](https://huggingface.co/dchaplinsky/punctuation_uk_bert) | token classification (NeMo) |
| punctuator2 | TODO: checkpoint / training data | bidirectional RNN with attention |

² Run locally via Ollama on one NVIDIA RTX 4090 Laptop GPU (16 GB), 4-bit
quantization, temperature 0, fixed instruction prompt, no in-prompt examples.
Invalid outputs (truncated, unpunctuated, or with low word overlap; 3 of 498
windows) were re-requested at temperature 0.3–0.4. Prompt: TODO (link to file).

## Versioning

Gold files are never changed silently. Any correction produces a new release
with a changelog entry; results are always reported with the release tag.

Releases are tagged for the whole repository; each subset is cited with the
release that introduced or last changed it.

- `v1.0` — UkrPunct-BRUK: Track A, four sets (this release).

## License

- **Data** (`data/bruk/`): derived from the Brown Corpus of Ukrainian (BRUK)
  and distributed, like BRUK, under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Changes to the source texts: selection of sentences, ellipses replaced with
  periods, segmentation into windows / sentences, and, in the model input,
  lowercasing and removal of punctuation.
- **Code** (`src/`): TODO (MIT / Apache-2.0).

## Citation

If you use the test sets, please cite the dataset, the theses and BRUK:

```bibtex
@misc{seliukh2026ukrpunct,
  author    = {Seliukh, R. and Sazhok, M. and Yukhymenko, O.},
  title     = {{UkrPunct-BRUK}: a test set for {Ukrainian} punctuation and capitalization restoration},
  version   = {1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.TODO},
  url       = {https://github.com/rseliukh/ukrpunct}
}

@inproceedings{seliukh2026itia,
  author    = {Seliukh, R. and Sazhok, M. and Yukhymenko, O.},
  title     = {Punctuation and Capitalization Restoration for {Ukrainian}: Specialized Models vs. Large Language Models},
  booktitle = {Інформаційні технології і автоматизація -- 2026},
  address   = {Odesa, Ukraine},
  year      = {2026},
  pages     = {TODO}
}

@inproceedings{starko2023bruk,
  author    = {Starko, Vasyl and Rysin, Andriy},
  title     = {Creating a {POS} Gold Standard Corpus of Modern {Ukrainian}},
  booktitle = {Proceedings of the Second Ukrainian Natural Language Processing Workshop (UNLP)},
  address   = {Dubrovnik, Croatia},
  year      = {2023},
  pages     = {91--95}
}
```
