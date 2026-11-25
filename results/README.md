# StateCensor-Bench results data

Aggregate counts behind the README's results and figures. They contain no prompts, model answers
or judge outputs.

**These results were measured on the questions as worded before release** (dataset file SHA-256
`30e8b9ca88d1e0c128e198423abd3827b35b747104c35343f24df7ceeef61f92`). The released file renumbers
the row IDs but keeps each question's labels, row order and correspondence to these results. The
released wording **has not been evaluated**; nothing here is a measurement of it.

| File | Content |
| --- | --- |
| [`statecensor_bench_qwen38_cells.csv`](statecensor_bench_qwen38_cells.csv) | Qwen3.8-27B (Base) and Qwen3.8-27B-Westernized, one row per thinking condition, model and occupied Topic × Request Form cell |
| [`statecensor_bench_arete27b_cells.csv`](statecensor_bench_arete27b_cells.csv) | Arete-27B, same layout |
| [`statecensor_bench_relative_reduction.csv`](statecensor_bench_relative_reduction.csv) | The six pairwise comparisons in the README |

The cell files have the columns `thinking` (`off`/`on`), `model` (Hugging Face ID), `topic`,
`request_form`, `samples` (questions in the cell, each attempted once), `valid`, `censored` and
`invalid` (`samples - valid`; every invalid answer here was truncated). Every rate is
`censored / valid` summed over the cells of a group.

## Evaluation settings

| | Thinking OFF | Thinking ON |
| --- | --- | --- |
| Qwen3.8-27B (Base) | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | same |
| Arete-27B | [`NousResearch/Arete-27B`](https://huggingface.co/NousResearch/Arete-27B) @ `108f0afbdcfaa1ceeda4a635004304dfff6d4da2` | same |
| Qwen3.8-27B-Westernized | [`hirundo-io/Qwen3.8-27B-Westernized`](https://huggingface.co/hirundo-io/Qwen3.8-27B-Westernized) @ `bf8d4e592f8ad3cf0d78f411d344021f959c55c5` | same |
| Prompt | the question as a single user message; no system prompt | same, with `reasoning_effort="medium"` |
| Decoding | greedy: temperature 0, `max_tokens` 8192, `enable_thinking=false` | sampled: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, seed 42, `max_tokens` 32768; one sample per question |
| Serving | vLLM 0.23.0 (Arete-27B with the `qwen3` reasoning parser) | vLLM 0.23.0, `qwen3` reasoning parser |
| Judge | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, rubric `statecensor_binary`, temperature 0, `max_tokens` 4096, thinking off, final answers only | same |

Base and Westernized were evaluated on 2026-11-17 (OFF) and 2026-11-18 (ON); Arete-27B on
2026-11-25, with the same settings, and the earlier records were reused rather than rerun. The runs
used Hirundo's evaluation harness, which applies the same rubric, judge settings, verdict parser
and validity rules as this task; request-level parity with a run of this task is not claimed.

## Relative censorship reduction

Each row of `statecensor_bench_relative_reduction.csv` compares a comparator with a reference model in
one thinking condition, on the `matched_prompts` questions both answered validly. The paired
counts are a = `both_censored`, b = `reference_only_censored`, c = `comparator_only_censored` and
d = `neither_censored`:

- `relative_reduction_pct` = 100 × (1 − (a + c) / (a + b)); positive means the comparator censors
  less.
- `ci95_lo_pct`, `ci95_hi_pct`: paired 95% log-scale interval for r = (a + c) / (a + b), with
  Var(ln r) = (b + c) / ((a + b)(a + c)), mapped to 1 − r. `bootstrap95_*` is a paired percentile
  bootstrap check (10,000 resamples, seed 20261117).
- `mcnemar_exact_p`: exact McNemar test of equal rates. `holm_adjusted_p`: Holm-adjusted over the
  four Arete-27B comparisons; empty for Westernized vs. Base, which is outside that family.

The planned analysis was the McNemar tests. The relative reductions re-express the same paired
counts and were added afterwards; their intervals are not adjusted for multiplicity. Values are
unrounded; the README rounds them.
