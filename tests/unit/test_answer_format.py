from __future__ import annotations

from ragkit.services.prompts import NOT_FOUND_TOKEN, build_prompt, format_model_answer
from tests.unit.test_answering import ENG, build

_FOLIC = """
- A dosage of folic acid of $0.8 \\text{ mg/d}$ appears necessary [1].
- The median decreased by $2.7 \\text{ micromol/L}$ ($23\\%$) at $0.8 \\text{ mg/d}$ [1].
- Lower doses are unclear, while $1 \\text{ mg/d}$ and above is known [1].
- A $25\\%$ reduction is achieved by **0.4 mg/d** [2].
""".strip()


def test_format_model_answer_strips_latex_and_keeps_numbers() -> None:
    plain = format_model_answer(_FOLIC)
    assert "$" not in plain
    assert "\\text" not in plain
    assert "0.8 mg/d" in plain
    assert "2.7 micromol/L" in plain
    assert "23%" in plain
    assert "25%" in plain
    assert "**0.4 mg/d**" in plain
    assert plain.startswith("- A dosage")
    assert "[1]" in plain and "[2]" in plain


def test_format_model_answer_leaves_abstention_token() -> None:
    assert format_model_answer(f"  {NOT_FOUND_TOKEN}  ") == NOT_FOUND_TOKEN
    assert format_model_answer("Refunds take 14 days [1].") == "Refunds take 14 days [1]."


def test_prompt_asks_for_plain_markdown() -> None:
    prompt = build_prompt("dose?", [])
    assert "Never use LaTeX" in prompt
    assert "**0.8 mg/d**" in prompt
    assert "start with yes or no" in prompt


def test_stream_stores_plain_units() -> None:
    service, _llm = build(r"Refunds take $14 \text{ days}$ [1].")
    events = list(service.stream_traced("refund?", ENG))
    done = next(event["result"] for event in events if event["type"] == "done")
    assert done.answer.text == "Refunds take 14 days [1]."
    assert done.answer.abstained is False


def test_langgraph_stores_plain_units() -> None:
    service, _llm = build(r"Refunds take $14 \text{ days}$ [1].")
    result = service.run_traced("refund?", ENG)
    assert result.answer.text == "Refunds take 14 days [1]."
    assert result.trace.generation == "Refunds take 14 days [1]."
