"""Consistency checks for the published StateCensor-Bench results and their figures.

The aggregate CSV must reproduce the verified headline counts and relative
reductions and match the dataset's group sizes; the README must show the figures.
"""

import csv
import importlib
import math
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CELLS = (RESULTS / "statecensor_bench_qwen38_cells.csv", RESULTS / "statecensor_bench_arete27b_cells.csv")
RELATIVE_REDUCTION = RESULTS / "statecensor_bench_relative_reduction.csv"
BASE, ARETE, WEST = (
    "Qwen/Qwen3.8-27B",
    "NousResearch/Arete-27B",
    "hirundo-io/Qwen3.8-27B-Westernized",
)
HEADLINE = {
    ("off", BASE): (422, 500),
    ("off", ARETE): (218, 500),
    ("off", WEST): (25, 499),
    ("on", BASE): (242, 500),
    ("on", ARETE): (71, 500),
    ("on", WEST): (15, 500),
}
"""Verified (censored, valid) per thinking condition and model."""
PAIRS = {(BASE, ARETE), (BASE, WEST), (ARETE, WEST)}
"""(reference, comparator) model pairs compared in each thinking condition."""
VERIFIED_REDUCTIONS = {
    ("off", BASE, ARETE): "**48.3%** [43.3, 52.9], n = 500",
    ("off", BASE, WEST): "**94.1%** [91.3, 95.9], n = 499",
    ("off", ARETE, WEST): "**88.5%** [83.1, 92.2], n = 499",
    ("on", BASE, ARETE): "**70.7%** [64.1, 76.0], n = 500",
    ("on", BASE, WEST): "**93.8%** [89.8, 96.3], n = 500",
    ("on", ARETE, WEST): "**78.9%** [64.3, 87.5], n = 500",
}
"""The verified relative reductions, as published."""
EVALUATED_JSONL_SHA256 = (
    "30e8b9ca88d1e0c128e198423abd3827b35b747104c35343f24df7ceeef61f92"
)
"""SHA-256 of the dataset file the published results were measured on."""
LEGACY_NAMES = re.compile(r"ccpc|500[-_]v2|\bR8(?:-P)?\b", re.IGNORECASE)
"""Development names that must not appear in public-facing text."""


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _cells() -> list[dict[str, str]]:
    return [cell for path in CELLS for cell in _read(path)]


def _sum(field: str, *keys: str) -> Counter[tuple[str, ...]]:
    sums: Counter[tuple[str, ...]] = Counter()
    for cell in _cells():
        sums[tuple(cell[key] for key in keys)] += int(cell[field])
    return sums


GROUP_SIZES = {
    "topic": {
        "ECONOMY_LABOR_INEQUALITY": 37,
        "ENVIRONMENT_INFRASTRUCTURE_LAND": 37,
        "GOVERNANCE_LAW_POLITICAL_INSTITUTIONS": 37,
        "HUMAN_RIGHTS_CIVIL_SOCIETY": 37,
        "INFORMATION_CONTROL_CENSORSHIP": 37,
        "INTERNATIONAL_RELATIONS_TRADE_POLICY": 37,
        "MILITARY_INTELLIGENCE_SECURITY": 37,
        "OTHER_PRC_POLITICAL_HISTORY": 37,
        "POLITICAL_MEDIA_ARTS_STATE_SYMBOLS": 37,
        "PUBLIC_HEALTH_BIOMEDICINE": 37,
        "RELIGION_BELIEF": 36,
        "TECHNOLOGY_INDUSTRIAL_CAPABILITY": 21,
        "TERRITORIAL_SOVEREIGNTY_AUTONOMY": 37,
        "TIANANMEN_1989": 36,
    },
    "request_form": {
        "ADVOCACY_ORGANIZING_CHANGE_STRATEGY": 93,
        "ANALYSIS_EVALUATION_COMPARISON": 84,
        "CONTENT_GENERATION_DRAFTING": 82,
        "EVASION_MANIPULATION_COVERT_TACTICS": 35,
        "INFORMATION_EXPLANATION": 84,
        "RESOURCE_SERVICE_ENTITY_RECOMMENDATION": 60,
        "RESTRICTED_SOURCE_DATA_ACCESS": 2,
        "TECHNICAL_IMPLEMENTATION_GUIDANCE": 60,
    },
}
"""Rows per Topic and per Request Form in the dataset."""


def test_headline_counts_match_the_verified_results() -> None:
    censored = _sum("censored", "thinking", "model")
    valid = _sum("valid", "thinking", "model")
    samples = _sum("samples", "thinking", "model")
    assert {key: (censored[key], valid[key]) for key in HEADLINE} == HEADLINE
    assert set(samples.values()) == {500} and len(samples) == len(HEADLINE)


def test_each_cells_file_holds_its_own_models() -> None:
    qwen, arete = (_read(path) for path in CELLS)
    assert {cell["model"] for cell in qwen} == {BASE, WEST}
    assert {cell["model"] for cell in arete} == {ARETE}


def test_cells_are_internally_consistent() -> None:
    cells = _cells()
    keys = [(c["thinking"], c["model"], c["topic"], c["request_form"]) for c in cells]
    assert len(keys) == len(set(keys)) == len(HEADLINE) * 95
    occupied = {key[2:] for key in keys}
    assert len(occupied) == 95
    for cell in cells:
        samples, valid, censored, invalid = (
            int(cell[field]) for field in ("samples", "valid", "censored", "invalid")
        )
        assert 0 <= censored <= valid <= samples and invalid == samples - valid


@pytest.mark.parametrize("field", ["topic", "request_form"])
def test_group_sizes_match_the_dataset(field: str) -> None:
    expected = GROUP_SIZES[field]
    assert sum(expected.values()) == 500
    sizes = _sum("samples", "thinking", "model", field)
    for thinking, model in HEADLINE:
        found = {key[2]: n for key, n in sizes.items() if key[:2] == (thinking, model)}
        assert found == expected


def test_readme_shows_existing_figures() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    figures = re.findall(r"\]\((figures/[\w.]+\.png)\)", readme)
    assert len(figures) == 5
    assert all((ROOT / figure).is_file() for figure in figures)


def test_results_stay_bound_to_the_evaluated_file_not_the_task_pin() -> None:
    task_module = importlib.import_module("statecensor_bench.statecensor_bench")
    provenance = (ROOT / "results" / "README.md").read_text(encoding="utf-8")
    assert EVALUATED_JSONL_SHA256 in provenance
    assert task_module.DATASET_JSONL_SHA256 != EVALUATED_JSONL_SHA256
    assert "has not been evaluated" in provenance


def test_public_text_uses_statecensor_bench_naming() -> None:
    """No development name survives in public text or file names."""
    paths = [
        ROOT / "README.md",
        ROOT / "results" / "README.md",
        ROOT / "pyproject.toml",
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
        *sorted((ROOT / "results").glob("*.csv")),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not LEGACY_NAMES.search(text), path.relative_to(ROOT)
    skip = {".git", ".venv", ".pytest_cache", ".ruff_cache"}
    names = [p.name for p in ROOT.rglob("*") if not skip & set(p.relative_to(ROOT).parts)]
    assert not any(LEGACY_NAMES.search(name) for name in names)


def test_relative_reductions_are_consistent_with_their_counts() -> None:
    rows = _read(RELATIVE_REDUCTION)
    keys = {(r["thinking"], r["reference_model"], r["comparator_model"]) for r in rows}
    assert len(rows) == 6
    assert keys == {(thinking, *pair) for thinking in ("off", "on") for pair in PAIRS}
    for row in rows:
        a, b, c, d = (
            int(row[f"{name}_censored"])
            for name in ("both", "reference_only", "comparator_only", "neither")
        )
        assert a + b + c + d == int(row["matched_prompts"])
        assert (a + b, a + c) == (
            int(row["reference_censored"]),
            int(row["comparator_censored"]),
        )
        ratio = (a + c) / (a + b)
        assert math.isclose(float(row["relative_reduction_pct"]), 100 * (1 - ratio), abs_tol=1e-3)
        half_width = 1.959964 * math.sqrt((b + c) / ((a + b) * (a + c)))
        low, high = (100 * (1 - ratio * math.exp(s * half_width)) for s in (1, -1))
        assert math.isclose(float(row["ci95_lo_pct"]), low, abs_tol=1e-3)
        assert math.isclose(float(row["ci95_hi_pct"]), high, abs_tol=1e-3)
        assert float(row["mcnemar_exact_p"]) < 1e-10
        in_family = row["comparator_model"] == ARETE or row["reference_model"] == ARETE
        assert bool(row["holm_adjusted_p"]) == in_family


def test_matched_counts_agree_with_the_headline_counts() -> None:
    for row in _read(RELATIVE_REDUCTION):
        for role in ("reference", "comparator"):
            censored, valid = HEADLINE[(row["thinking"], row[f"{role}_model"])]
            matched = int(row["matched_prompts"])
            assert matched <= valid
            if matched == valid:
                assert int(row[f"{role}_censored"]) == censored


def test_relative_reductions_match_the_verified_values() -> None:
    rows = {
        (r["thinking"], r["reference_model"], r["comparator_model"]): r
        for r in _read(RELATIVE_REDUCTION)
    }
    for key, quoted in VERIFIED_REDUCTIONS.items():
        row = rows[key]
        expected = (
            f"**{float(row['relative_reduction_pct']):.1f}%** "
            f"[{float(row['ci95_lo_pct']):.1f}, {float(row['ci95_hi_pct']):.1f}], "
            f"n = {row['matched_prompts']}"
        )
        assert quoted == expected
