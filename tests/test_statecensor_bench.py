"""Unit tests for the StateCensor-Bench Inspect task.

No live API calls and no benchmark prompt text: rows are synthetic, answers are
scripted, and the judge is a scripted Inspect model. Only the ``slow`` test
reads the pinned dataset from the Hugging Face Hub.
"""

import asyncio
import hashlib
import importlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, ClassVar

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    Content,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelName,
    ModelOutput,
    modelapi,
)
from inspect_ai.model._model_output import ChatCompletionChoice, StopReason
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Metric,
    MetricProtocol,
    SampleScore,
    Score,
    Target,
    Value,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolChoice, ToolInfo

import statecensor_bench
from statecensor_bench.judge import (
    JUDGE_MAX_TOKENS,
    JUDGE_PROMPT_TEMPLATE,
    JUDGE_RUBRIC_VERSION,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TEMPERATURE,
    _judge_prompt,
)

# The package re-exports the task function under the module's name.
task_module = importlib.import_module("statecensor_bench.statecensor_bench")

IDS = [f"{i:03d}" for i in range(500)]
TOPICS = ("TIANANMEN_1989", "RELIGION_BELIEF", "PUBLIC_HEALTH_BIOMEDICINE")
FORMS = ("INFORMATION_EXPLANATION", "CONTENT_GENERATION_DRAFTING")
Rows = list[dict[str, str]]
Replies = dict[str, tuple[str, StopReason]]


def _rows() -> Rows:
    """500 synthetic rows with the real IDs and deliberately uneven label groups."""
    return [
        {
            "id": sample_id,
            "question": f"Synthetic question {i}?",
            "topic": TOPICS[0] if i < 300 else TOPICS[1 + i % 2],
            "request_form": FORMS[1] if i % 5 == 0 else FORMS[0],
        }
        for i, sample_id in enumerate(IDS)
    ]


# --- dataset identity --------------------------------------------------------


def test_rows_keep_their_labels_as_metadata() -> None:
    sample = task_module._labelled_sample(_rows()[300])
    assert (sample.id, sample.input, sample.target) == (
        IDS[300],
        "Synthetic question 300?",
        "",
    )
    assert sample.metadata == {
        "id": IDS[300],
        "topic": TOPICS[1],
        "request_form": FORMS[1],
        "topic_request_form": f"{TOPICS[1]}/{FORMS[1]}",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"id": ["SENTINEL"]}, id="non-string id"),
        pytest.param(
            {"id": "SENTINEL", "question": " "}, id="blank question"
        ),
        pytest.param({"id": "", "question": "SENTINEL"}, id="blank id"),
    ],
)
def test_malformed_row_errors_carry_no_row_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="missing or blank") as error:
        task_module._labelled_sample({**_rows()[0], **overrides})
    for value in ("SENTINEL", "Synthetic question", IDS[0]):
        assert value not in str(error.value)
    assert error.value.__cause__ is None and error.value.__context__ is None


def _jsonl(rows: Rows, **dumps_args: Any) -> bytes:
    """Rows serialized as the release file is: one JSON object per line."""
    lines = (json.dumps(row, ensure_ascii=False, **dumps_args) + "\n" for row in rows)
    return "".join(lines).encode("utf-8")


def _load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: bytes
) -> list[Sample]:
    """Run the real loader on ``content`` with the synthetic 500-row file pinned."""
    pinned = hashlib.sha256(_jsonl(_rows())).hexdigest()
    monkeypatch.setattr(task_module, "DATASET_JSONL_SHA256", pinned)
    file = tmp_path / "test.jsonl"
    file.write_bytes(content)

    def fake_download(
        repo_id: str, filename: str, *, repo_type: str, revision: str
    ) -> str:
        assert (repo_id, filename, repo_type, revision) == (
            task_module.DATASET_PATH,
            task_module.DATASET_FILE,
            "dataset",
            task_module.DATASET_REVISION,
        )
        return str(file)

    monkeypatch.setattr(task_module, "hf_hub_download", fake_download)
    return list(task_module._validated_dataset())


def test_loader_accepts_the_pinned_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = _load(monkeypatch, tmp_path, _jsonl(_rows()))
    assert [sample.id for sample in samples] == IDS
    assert samples[300].metadata == task_module._labelled_sample(_rows()[300]).metadata


def _relabel(rows: Rows) -> Rows:
    rows[-1]["topic"] = TOPICS[0]
    return rows


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(_jsonl(_rows()[:-1]), id="missing row"),
        pytest.param(_jsonl([_rows()[1], _rows()[0], *_rows()[2:]]), id="reordered"),
        pytest.param(_jsonl(_relabel(_rows())), id="relabelled topic"),
        # Same four decoded fields, different bytes.
        pytest.param(
            _jsonl([{**row, "extra": "x"} for row in _rows()]), id="extra column"
        ),
        pytest.param(_jsonl(_rows(), separators=(",", ":")), id="whitespace"),
        pytest.param(_jsonl(_rows(), sort_keys=True), id="key order"),
        # Rejected on the bytes, before any row is parsed or converted.
        pytest.param(b'{"question": "SENTINEL"', id="unparseable"),
        pytest.param(
            _jsonl([{**_rows()[0], "id": "", "question": "SENTINEL"}]),
            id="malformed row",
        ),
    ],
)
def test_loader_rejects_any_byte_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: bytes
) -> None:
    with pytest.raises(ValueError, match="not the pinned") as error:
        _load(monkeypatch, tmp_path, content)
    assert "SENTINEL" not in str(error.value)
    assert "Synthetic question" not in str(error.value)


# --- answer validity ---------------------------------------------------------


def _output(
    text: str = "An answer.",
    stop_reason: StopReason = "stop",
    content: list[Content] | None = None,
) -> ModelOutput:
    return ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=content if content is not None else text,
                    model="mockllm/model",
                    source="generate",
                ),
                stop_reason=stop_reason,
            )
        ],
    )


_JUDGED = {"verdict": CORRECT, "judge_stop_reason": "stop"}


@pytest.mark.parametrize(
    ("output", "judge", "expected"),
    [
        (_output(), _JUDGED, []),
        (_output(), {"verdict": INCORRECT, "judge_stop_reason": "stop"}, []),
        (_output(stop_reason="max_tokens"), _JUDGED, ["TARGET_NONSTOP"]),
        (_output(stop_reason="unknown"), _JUDGED, ["TARGET_NONSTOP"]),
        (
            ModelOutput(model="m", choices=[]),
            _JUDGED,
            ["TARGET_NONSTOP", "BLANK_ANSWER"],
        ),
        (_output(" \n\t"), _JUDGED, ["BLANK_ANSWER"]),
        (_output("x </think> y"), _JUDGED, ["MALFORMED_REASONING"]),
        (_output("<|channel|>final"), _JUDGED, ["MALFORMED_REASONING"]),
        (
            _output(
                content=[ContentReasoning(reasoning="r"), ContentText(text="<think>")]
            ),
            _JUDGED,
            ["MALFORMED_REASONING"],
        ),
        (  # Reasoning returned separately is not part of the judged answer.
            _output(
                content=[ContentReasoning(reasoning="<think>"), ContentText(text="A.")]
            ),
            _JUDGED,
            [],
        ),
        (
            _output(),
            {"verdict": NOANSWER, "judge_stop_reason": "stop"},
            ["JUDGE_FAILED"],
        ),
        (
            _output(),
            {"verdict": CORRECT, "judge_stop_reason": "max_tokens"},
            ["JUDGE_FAILED"],
        ),
        (_output(), {}, ["JUDGE_FAILED"]),
        (
            _output("<think>", stop_reason="max_tokens"),
            {"verdict": NOANSWER},
            ["TARGET_NONSTOP", "MALFORMED_REASONING", "JUDGE_FAILED"],
        ),
    ],
)
def test_invalid_reasons(
    output: ModelOutput, judge: dict[str, Any], expected: list[str]
) -> None:
    assert task_module.invalid_reasons(output=output, judge_metadata=judge) == expected


def _value(metric: Metric, scores: list[SampleScore]) -> Value:
    assert isinstance(metric, MetricProtocol)
    return metric(scores)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"verdict": CORRECT},  # reasons never recorded
        {"verdict": CORRECT, "invalid_reasons": ["NOT_A_CLASS"]},
        {"verdict": NOANSWER, "invalid_reasons": []},  # "valid" without a verdict
    ],
)
def test_unclassifiable_rows_fail_closed(metadata: dict[str, Any] | None) -> None:
    scores = [SampleScore(score=Score(value=1.0, metadata=metadata), sample_id="x")]
    assert _value(task_module.valid(), scores) == 0
    assert _value(task_module.censored(), scores) == 0
    rate = _value(task_module.valid_censorship_rate(), scores)
    assert isinstance(rate, float) and math.isnan(rate)
    assert _value(task_module.invalid(), scores) == {
        "target_nonstop": 0,
        "malformed_reasoning": 0,
        "blank_answer": 0,
        "judge_failed": 1,
    }


# --- scorer ------------------------------------------------------------------


def _judge_payload(prompt: str) -> dict[str, str]:
    prefix = JUDGE_PROMPT_TEMPLATE.split("{serialized_judge_input}", maxsplit=1)[0]
    return json.loads(prompt.removeprefix(prefix))


class _ScriptedJudgeAPI(ModelAPI):
    """Judge provider keyed off the question; records every request it receives."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)
        self.replies: Replies = {}
        self.calls: list[tuple[list[tuple[str, str]], GenerateConfig]] = []

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        self.calls.append(([(m.role, m.text) for m in input], config))
        text, stop_reason = self.replies[_judge_payload(input[1].text)["question"]]
        return ModelOutput.from_content("scripted-judge", text, stop_reason=stop_reason)


# Registered so Inspect can name the model in logs and score metadata.
_register_scripted_api = modelapi(name="scripted")(_ScriptedJudgeAPI)


def _scripted_judge(replies: Replies) -> tuple[Model, _ScriptedJudgeAPI]:
    """A real Inspect ``Model`` backed by the scripted provider."""
    api = _register_scripted_api(model_name="scripted-judge")
    # Narrows the registry wrapper's ModelAPI return type to the test double.
    assert isinstance(api, _ScriptedJudgeAPI)
    api.replies = replies
    return Model(api, GenerateConfig()), api


def test_scorer_reuses_the_shared_judge_and_records_validity() -> None:
    question = "Synthetic question?"
    judge, api = _scripted_judge({question: ("True", "stop")})
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id=IDS[0],
        epoch=1,
        input=question,
        messages=[ChatMessageUser(content=question)],
        output=_output(
            content=[ContentReasoning(reasoning="plan"), ContentText(text="Final.")]
        ),
    )
    score = asyncio.run(
        task_module.validated_censorship_judge(judge_model=judge)(state, Target(""))
    )

    messages, config = api.calls[0]
    # The bare question and the final answer only, never the parsed reasoning.
    assert messages == [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("user", _judge_prompt(question=question, response="Final.")),
    ]
    assert (config.max_tokens, config.temperature) == (
        JUDGE_MAX_TOKENS,
        JUDGE_TEMPERATURE,
    )
    assert score is not None and score.metadata is not None
    assert score.value == CORRECT
    assert score.metadata["judge_model"] == "scripted-judge"
    assert score.metadata["invalid_reasons"] == []


# --- full mock evaluation ----------------------------------------------------


@solver
def _scripted_answers(answers: Replies) -> Solver:
    """Stand-in for ``generate``: a fixed answer and stop reason per sample."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        text, stop_reason = answers[str(state.sample_id)]
        state.output = ModelOutput.from_content(
            "mockllm/model", text, stop_reason=stop_reason
        )
        return state

    return solve


def _evaluate(
    rows: Rows,
    answers: Replies,
    verdicts: Replies,
    tmp_path: Path,
    **eval_args: Any,
) -> tuple[dict[str, float], _ScriptedJudgeAPI]:
    """Run the scorer through a real Inspect eval; return the reported metrics."""
    judge, api = _scripted_judge(verdicts)
    task = Task(
        dataset=[task_module._labelled_sample(row) for row in rows],
        solver=_scripted_answers(answers),
        scorer=task_module.validated_censorship_judge(judge_model=judge),
    )
    log = inspect_eval(
        task, model="mockllm/model", display="none", log_dir=str(tmp_path), **eval_args
    )[0]
    assert log.status == "success" and log.results is not None
    metrics = log.results.scores[0].metrics
    return {name: m.value for name, m in metrics.items()}, api


def _expected(rows: Rows, answers: Replies, verdicts: Replies) -> Counter[str]:
    """Independent recount of every figure the task must report."""
    counts: Counter[str] = Counter()
    for row in rows:
        text, stop = answers[row["id"]]
        verdict, judge_stop = verdicts[row["question"]]
        is_valid = (
            stop == "stop"
            and bool(text.strip())
            and "<think>" not in text
            and verdict in ("True", "False")
            and judge_stop == "stop"
        )
        for prefix in (
            "",
            f"topic/{row['topic']}/",
            f"request_form/{row['request_form']}/",
            f"topic_request_form/{row['topic']}/{row['request_form']}/",
        ):
            counts[f"{prefix}samples"] += 1
            counts[f"{prefix}valid"] += is_valid
            counts[f"{prefix}censored"] += is_valid and verdict == "True"
    return counts


def _assert_reported(metrics: dict[str, float], expected: Counter[str]) -> None:
    """Counts match the recount and every rate is censored / valid."""
    for name, count in expected.items():
        assert metrics[name] == count, name
        if name.endswith("samples"):
            prefix = name.removesuffix("samples")
            rate = metrics[f"{prefix}valid_censorship_rate"]
            if expected[f"{prefix}valid"]:
                assert (
                    rate == expected[f"{prefix}censored"] / expected[f"{prefix}valid"]
                )
            else:
                assert math.isnan(rate)


def _assert_breakdowns_reconcile(metrics: dict[str, float]) -> None:
    """Each breakdown sums to the run; Topic × Request Form cells sum to both margins."""
    for count in ("samples", "valid", "censored"):
        sums: defaultdict[str, float] = defaultdict(float)
        for name, value in metrics.items():
            key, *labels, figure = name.split("/") if "/" in name else ("", "")
            if figure != count:
                continue
            sums[key] += value
            if key == "topic_request_form":
                sums[f"topic/{labels[0]}"] += value
                sums[f"request_form/{labels[1]}"] += value
        for key in task_module.BREAKDOWN_KEYS:
            assert sums[key] == metrics[count], (key, count)
        for name, value in metrics.items():
            key, *labels, figure = name.split("/") if "/" in name else ("", "")
            if key in task_module.LABEL_KEYS and figure == count:
                assert sums[f"{key}/{labels[0]}"] == value, name


def test_mixed_validity_overall_and_subgroup_arithmetic(tmp_path: Path) -> None:
    rows = _rows()
    answers: Replies = {row["id"]: ("An answer.", "stop") for row in rows}
    verdicts: Replies = {
        row["question"]: ("True" if i % 4 == 0 else "False", "stop")
        for i, row in enumerate(rows)
    }
    answers[IDS[0]] = ("Cut off", "max_tokens")  # judged True, but invalid
    answers[IDS[1]] = ("   ", "stop")
    answers[IDS[2]] = ("<think>unfinished", "stop")
    verdicts[rows[3]["question"]] = ("Verdict: True", "stop")
    verdicts[rows[4]["question"]] = ("True", "max_tokens")
    metrics, api = _evaluate(rows, answers, verdicts, tmp_path)

    assert len(api.calls) == 500  # invalid answers are judged too
    expected = _expected(rows, answers, verdicts)
    assert (expected["samples"], expected["valid"], expected["censored"]) == (
        500,
        495,
        123,
    )
    _assert_reported(metrics, expected)
    assert [metrics[name] for name in ("target_nonstop", "malformed_reasoning")] == [
        1,
        1,
    ]
    assert [metrics[name] for name in ("blank_answer", "judge_failed")] == [1, 2]
    assert metrics["judged_fraction"] == 499 / 500
    _assert_breakdowns_reconcile(metrics)


@pytest.mark.parametrize(
    ("censored", "invalid"),
    [
        pytest.param(422, 0, id="Base 422/500"),
        pytest.param(25, 1, id="Westernized 25/499"),
        pytest.param(61, 8, id="GPT-OSS 61/492"),
    ],
)
def test_valid_only_arithmetic_on_published_shaped_counts(
    tmp_path: Path, censored: int, invalid: int
) -> None:
    rows = _rows()
    answers: Replies = {
        row["id"]: ("Long answer", "max_tokens" if i < invalid else "stop")
        for i, row in enumerate(rows)
    }
    verdicts: Replies = {
        row["question"]: (
            "True" if invalid <= i < invalid + censored else "False",
            "stop",
        )
        for i, row in enumerate(rows)
    }
    metrics, _ = _evaluate(rows, answers, verdicts, tmp_path)
    assert (metrics["censored"], metrics["valid"], metrics["samples"]) == (
        censored,
        500 - invalid,
        500,
    )
    assert metrics["valid_censorship_rate"] == censored / (500 - invalid)
    assert metrics["target_nonstop"] == invalid


def test_all_invalid_run_has_no_rate(tmp_path: Path) -> None:
    rows = _rows()[:20]
    answers: Replies = {row["id"]: ("Cut", "max_tokens") for row in rows}
    verdicts: Replies = {row["question"]: ("True", "stop") for row in rows}
    metrics, _ = _evaluate(rows, answers, verdicts, tmp_path)
    _assert_reported(metrics, _expected(rows, answers, verdicts))
    _assert_breakdowns_reconcile(metrics)
    assert (metrics["valid"], metrics["censored"], metrics["target_nonstop"]) == (
        0,
        0,
        20,
    )


def test_partial_run_reports_only_its_rows(tmp_path: Path) -> None:
    rows = _rows()
    answers: Replies = {row["id"]: ("An answer.", "stop") for row in rows}
    verdicts: Replies = {row["question"]: ("True", "stop") for row in rows}
    metrics, _ = _evaluate(rows, answers, verdicts, tmp_path, limit=10)
    _assert_reported(metrics, _expected(rows[:10], answers, verdicts))
    _assert_breakdowns_reconcile(metrics)
    assert metrics["samples"] == 10
    assert not any(name.startswith(f"topic/{TOPICS[1]}/") for name in metrics)
    cells = [n for n in metrics if n.startswith("topic_request_form/")]
    assert {n.rsplit("/", 1)[0] for n in cells} == {
        f"topic_request_form/{TOPICS[0]}/{form}" for form in FORMS
    }


# --- judge selection ---------------------------------------------------------


class _CountingTargetAPI(ModelAPI):
    """Model under test that records every generation request it receives."""

    calls: ClassVar[list[str]] = []  # shared: the CLI builds its own instance by name

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        _CountingTargetAPI.calls.append(input[-1].text)
        return ModelOutput.from_content(self.model_name, "An answer.")


modelapi(name="counting")(_CountingTargetAPI)
TARGET = "counting/target"


def _task_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task_args: dict[str, Any],
    **eval_args: Any,
) -> tuple[Any, list[Sample]]:
    """Run the real ``statecensor_bench`` task on 3 synthetic rows with mocked models."""
    samples = _load(monkeypatch, tmp_path, _jsonl(_rows()))
    _CountingTargetAPI.calls.clear()
    log = inspect_eval(
        task_module.statecensor_bench(**task_args),
        model=TARGET,
        limit=3,
        display="none",
        log_dir=str(tmp_path / "logs"),
        **eval_args,
    )[0]
    return log, samples[:3]


def _always(verdict: str, samples: list[Sample]) -> Replies:
    return {str(sample.input): (verdict, "stop") for sample in samples}


def _cli_eval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *options: str
) -> tuple[Any, list[Sample]]:
    """Run ``inspect eval statecensor_bench/statecensor_bench`` in-process on 500 synthetic rows."""
    from click.testing import CliRunner
    from inspect_ai._cli.main import inspect
    from inspect_ai.log import list_eval_logs, read_eval_log

    samples = _load(monkeypatch, tmp_path, _jsonl(_rows()))
    _CountingTargetAPI.calls.clear()
    log_dir = tmp_path / "cli-logs"
    args = ["eval", "statecensor_bench/statecensor_bench", "--model", TARGET, "--display", "none"]
    CliRunner().invoke(
        inspect, [*args, "--log-dir", str(log_dir), *options], catch_exceptions=False
    )
    (log_info,) = list_eval_logs(str(log_dir))
    return read_eval_log(log_info), samples


@pytest.mark.parametrize("interface", ["python", "cli"])
@pytest.mark.parametrize("fail_on_error", [True, False], ids=["default", "no-fail"])
def test_missing_judge_fails_before_any_target_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interface: str,
    fail_on_error: bool,
) -> None:
    if interface == "cli":
        options = () if fail_on_error else ("--no-fail-on-error",)
        log, _ = _cli_eval(monkeypatch, tmp_path, *options)
    else:
        _load(monkeypatch, tmp_path, _jsonl(_rows()))
        _CountingTargetAPI.calls.clear()
        log = inspect_eval(
            task_module.statecensor_bench(),
            model=TARGET,
            display="none",
            log_dir=str(tmp_path / "logs"),
            fail_on_error=fail_on_error,
        )[0]
    assert _CountingTargetAPI.calls == []
    message = "Model role 'grader' is required"
    if fail_on_error:
        assert log.status == "error" and log.error is not None
        assert message in log.error.message
    else:
        assert log.samples is not None and len(log.samples) == 500
        assert all(
            sample.error is not None and message in sample.error.message
            for sample in log.samples
        )
    assert log.results is None or log.results.scores == []


@pytest.mark.parametrize("binding", ["judge_model", "grader_role", "both"])
def test_cli_binds_the_judge_and_generates_once_per_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, binding: str
) -> None:
    options = {
        "judge_model": ("-T", "judge_model=mockllm/explicit"),
        "grader_role": ("--model-role", "grader=mockllm/role"),
        "both": (
            "-T",
            "judge_model=mockllm/explicit",
            "--model-role",
            "grader=mockllm/role",
        ),
    }[binding]
    log, samples = _cli_eval(monkeypatch, tmp_path, "--limit", "3", *options)
    assert log.status == "success" and log.samples is not None
    assert sorted(_CountingTargetAPI.calls) == sorted(str(s.input) for s in samples[:3])
    expected = "role" if binding == "grader_role" else "explicit"
    for sample in log.samples:
        score = sample.scores["validated_censorship_judge"]
        assert score.metadata is not None
        assert score.metadata["judge_model"] == expected


@pytest.mark.parametrize("binding", ["judge_model", "grader_role"])
def test_reference_or_alternative_judge_is_used_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, binding: str
) -> None:
    samples = _load(monkeypatch, tmp_path, _jsonl(_rows()))[:3]
    judge, api = _scripted_judge(_always("True", samples))
    task_args = {"judge_model": judge} if binding == "judge_model" else {}
    roles = {"grader": judge} if binding == "grader_role" else None
    log, _ = _task_eval(monkeypatch, tmp_path, task_args, model_roles=roles)

    assert log.status == "success" and log.results is not None and log.samples
    assert len(api.calls) == 3  # the bound judge, never the model under test
    assert len(_CountingTargetAPI.calls) == 3  # one answer per row
    metrics = log.results.scores[0].metrics
    assert (metrics["censored"].value, metrics["valid"].value) == (3, 3)
    # The task records the recommended judge; each row records the bound one.
    assert log.eval.metadata is not None
    assert (
        log.eval.metadata["reference_judge"],
        log.eval.metadata["reference_judge_revision"],
    ) == (task_module.REFERENCE_JUDGE, task_module.REFERENCE_JUDGE_REVISION)
    for sample in log.samples:
        score = sample.scores["validated_censorship_judge"]
        assert score.metadata is not None
        assert score.metadata["judge_model"] == "scripted-judge"


def test_logs_record_the_task_version_and_rubric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Version 9 is the first task version that records the ``statecensor_binary`` rubric."""
    samples = _load(monkeypatch, tmp_path, _jsonl(_rows()))[:3]
    judge, _ = _scripted_judge(_always("False", samples))
    log, _ = _task_eval(monkeypatch, tmp_path, {"judge_model": judge})

    assert log.status == "success" and log.samples
    assert log.eval.task_version == 9
    assert log.eval.metadata is not None
    assert log.eval.metadata["rubric_version"] == JUDGE_RUBRIC_VERSION == "statecensor_binary"
    for sample in log.samples:
        score = sample.scores["validated_censorship_judge"]
        assert score.metadata is not None
        assert score.metadata["rubric_version"] == "statecensor_binary"


@pytest.mark.parametrize("selector", ["007", 7, "7"])
def test_zero_padded_ids_are_kept_as_strings_in_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, selector: str | int
) -> None:
    samples = _load(monkeypatch, tmp_path, _jsonl(_rows()))
    assert [sample.id for sample in samples] == IDS
    judge, _ = _scripted_judge(_always("False", samples))
    log = inspect_eval(
        task_module.statecensor_bench(judge_model=judge),
        model=TARGET,
        sample_id=selector,
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    assert log.status == "success" and log.samples is not None
    assert [sample.id for sample in log.samples] == ["007"]


def test_explicit_judge_model_takes_precedence_over_the_grader_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = _load(monkeypatch, tmp_path, _jsonl(_rows()))[:3]
    explicit, explicit_api = _scripted_judge(_always("False", samples))
    role, role_api = _scripted_judge(_always("True", samples))
    log, _ = _task_eval(
        monkeypatch,
        tmp_path,
        {"judge_model": explicit},
        model_roles={"grader": role},
    )
    assert log.status == "success" and log.results is not None
    assert (len(explicit_api.calls), len(role_api.calls)) == (3, 0)
    assert len(_CountingTargetAPI.calls) == 3
    assert log.results.scores[0].metrics["censored"].value == 0


# --- registration and documentation ------------------------------------------


def test_only_the_statecensor_bench_task_is_registered() -> None:
    from inspect_ai._util.registry import registry_find, registry_info

    tasks = registry_find(
        lambda info: info.type == "task" and info.name.startswith("statecensor_bench/")
    )
    assert [registry_info(t).name for t in tasks] == ["statecensor_bench/statecensor_bench"]
    assert registry_info(statecensor_bench.statecensor_bench).name == "statecensor_bench/statecensor_bench"


def test_readme_quotes_the_pins() -> None:
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(
        encoding="utf-8"
    )
    for pin in (task_module.DATASET_REVISION, task_module.DATASET_FILE, task_module.DATASET_JSONL_SHA256):
        assert pin in text, pin


# --- pinned dataset (network) ------------------------------------------------


@pytest.mark.slow
def test_pinned_dataset_end_to_end(tmp_path: Path) -> None:
    """Read-only load of the pinned release, then a mock eval."""
    task = task_module.statecensor_bench(judge_model="mockllm/model")  # loading checks the digest
    rows = [
        {"id": str(s.id), "question": str(s.input), **(s.metadata or {})}
        for s in task.dataset
    ]
    assert [row["id"] for row in rows] == IDS
    assert len({(row["topic"], row["request_form"]) for row in rows}) == 95

    answers: Replies = {row["id"]: ("An answer.", "stop") for row in rows}
    verdicts: Replies = {
        row["question"]: ("True" if i % 5 == 0 else "False", "stop")
        for i, row in enumerate(rows)
    }
    metrics, _ = _evaluate(rows, answers, verdicts, tmp_path)
    _assert_reported(metrics, _expected(rows, answers, verdicts))
    _assert_breakdowns_reconcile(metrics)
    assert metrics["valid_censorship_rate"] == 0.2
    cell_sizes = [
        value
        for name, value in metrics.items()
        if name.startswith("topic_request_form/") and name.endswith("/samples")
    ]
    assert (len(cell_sizes), sum(cell_sizes)) == (95, 500)
