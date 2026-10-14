"""LLM judge for the two metrics deterministic scorers cannot measure: faithfulness and graded correctness."""

from typing import Any

from oagbench.generate import GoldQuestion
from oagbench.llm import LLMError, LLMProvider

JUDGE_SYSTEM = """You grade an answer produced by a retrieval-augmented system.

Return:
- "faithfulness": share (0 to 1) of the claims in the answer that are supported by the retrieved context.
  An abstention ("NO_ANSWER") with no claims scores 1.
- "correctness": how well the answer matches the reference answer (0 to 1). Penalise missing and extra items.
  If the reference is "no answer exists", an abstention scores 1 and any concrete answer scores 0.
- "reason": one short sentence.
Judge only against the reference and the context, not your own knowledge."""

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["faithfulness", "correctness", "reason"],
    "properties": {
        "faithfulness": {"type": "number"},
        "correctness": {"type": "number"},
        "reason": {"type": "string"},
    },
}


def judge_response(provider: LLMProvider, model: str | None, q: GoldQuestion, response: dict[str, Any]) -> dict:
    context = "\n".join(f"- (doc: {i['source_doc']}) {i['content']}" for i in response["context"]["items"])
    reference = "no answer exists" if q.answer_type == "no_answer" else "; ".join(q.answer)
    prompt = (f"Question: {q.question}\nReference answer: {reference}\n\nRetrieved context:\n{context or '(none)'}"
              f"\n\nAnswer to grade: {response['answer']}\nAnswer items: {response['items']}")
    try:
        result = provider.complete(system=JUDGE_SYSTEM, prompt=prompt, schema=JUDGE_SCHEMA, max_tokens=4000,
                                   model=model)
    except LLMError as exc:
        return {"judge_error": str(exc)}
    data = result.data or {}
    clamp = lambda v: min(1.0, max(0.0, float(v)))  # noqa: E731
    return {
        "judge_faithfulness": clamp(data.get("faithfulness", 0)),
        "judge_correctness": clamp(data.get("correctness", 0)),
        "judge_reason": str(data.get("reason", "")),
        "judge_call": result.trace("judge"),
    }
