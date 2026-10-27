from __future__ import annotations

import re
from collections.abc import Sequence

from ragkit.domain.models import ScoredChunk

NOT_FOUND_TOKEN = "NOT_FOUND"

_TEMPLATE = """You are a precise enterprise and scientific research assistant.
Answer the question using ONLY the numbered evidence below.

Format the answer as readable markdown:
- When there is more than one finding, use a markdown bullet list. One finding per bullet.
- Bold the key number or conclusion in each bullet, for example **0.8 mg/d**.
- Write numbers and units in plain text: 0.8 mg/d, 2.7 µmol/L, 23%, ≥ 4000 g.
- Never use LaTeX, dollar-math, or commands such as \\text, \\mathrm, or \\%.
- End every sentence or bullet with a citation like [1] or [2].
- Do not use outside knowledge. If the question asks about a specific entity, substance,
  or policy not in the evidence, do not substitute a different one—reply exactly {token}.
- If the evidence is insufficient, reply exactly {token}.

Answer every part of the question, then stop.
- If the question names two subjects, give each subject its own sentence and name it.
- If the question asks whether two figures are the same, start with yes or no, then give
  both figures and their units.
- Use a number only if it appears in the cited evidence.

Evidence:
{evidence}

Question: {question}
Answer:"""

_MATH = re.compile(r"\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$")
_TEXT_CMD = re.compile(
    r"\\(?:text|mathrm|mathbf|operatorname|textrm|textit)\{([^{}]*)\}"
)
_COMMAND = re.compile(r"\\[a-zA-Z]+")
_LATEX_WORDS = (
    (r"\%", "%"),
    (r"\geq", "≥"),
    (r"\ge", "≥"),
    (r"\leq", "≤"),
    (r"\le", "≤"),
    (r"\times", "x"),
    (r"\pm", "±"),
    (r"\cdot", "·"),
    (r"\mu", "µ"),
    (r"\alpha", "alpha"),
    (r"\beta", "beta"),
    (r"\_", "_"),
)


def _plain_math(body: str) -> str:
    """Turn one LaTeX math span into plain text, keeping the numbers."""
    previous = None
    while previous != body:
        previous = body
        body = _TEXT_CMD.sub(r"\1", body)
    for source, target in _LATEX_WORDS:
        body = body.replace(source, target)
    body = _COMMAND.sub("", body)
    body = body.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]{2,}", " ", body).strip()


def format_model_answer(text: str) -> str:
    """Make a model completion readable: plain units, no `$...$` or `\\text{}`."""
    stripped = text.strip()
    if not stripped or stripped == NOT_FOUND_TOKEN or stripped.startswith(NOT_FOUND_TOKEN):
        return stripped

    def replace_math(match: re.Match[str]) -> str:
        inner = match.group(1) if match.group(1) is not None else match.group(2)
        return _plain_math(inner or "")

    cleaned = _MATH.sub(replace_math, stripped)
    if "\\" in cleaned:
        cleaned = _plain_math(cleaned)
    cleaned = cleaned.replace("$", "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_prompt(question: str, evidence: Sequence[ScoredChunk]) -> str:
    lines = "\n".join(f"[{i}] {s.chunk.text}" for i, s in enumerate(evidence, start=1))
    return _TEMPLATE.format(token=NOT_FOUND_TOKEN, evidence=lines, question=question)
