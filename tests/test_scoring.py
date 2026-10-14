import math

from oagbench.scoring import score_response


def _resp(items, answer="x", abstained=False, citations=()):
    return {"items": list(items), "answer": answer, "abstained": abstained, "citations": list(citations)}


def test_gold_answers_score_perfectly(corpus) -> None:
    g = corpus.canonical
    index = g.alias_index()
    for q in corpus.gold:
        if q.answer_type == "no_answer":
            resp = _resp([], "NO_ANSWER", abstained=True)
        else:
            resp = _resp(q.answer, " ".join(q.answer), citations=q.supporting_docs)
        scores = score_response(g, index, q, resp)
        assert scores["correct"] == 1.0, (q.qid, q.question, q.answer)
        if q.supporting_docs:
            assert scores["citation_recall"] == 1.0


def test_aliases_resolve_to_the_same_entity(corpus) -> None:
    g = corpus.canonical
    index = g.alias_index()
    q = next(q for q in corpus.gold if q.template == "owner")
    team = g.entities[q.answer_ids[0]]
    assert score_response(g, index, q, _resp([f"{team.name} team"]))["correct"] == 1.0


def test_partial_set_gets_partial_f1(corpus) -> None:
    g = corpus.canonical
    index = g.alias_index()
    q = next(q for q in corpus.gold if q.answer_type == "entities" and len(q.answer) >= 2)
    scores = score_response(g, index, q, _resp(q.answer[:1]))
    assert scores["correct"] == 0.0
    assert scores["precision"] == 1.0
    assert 0 < scores["f1"] < 1


def test_hallucinating_on_unanswerable_scores_zero(corpus) -> None:
    g = corpus.canonical
    q = next(q for q in corpus.gold if q.answer_type == "no_answer")
    scores = score_response(g, g.alias_index(), q, _resp(["Ledger"], "Ledger owns it."))
    assert scores["correct"] == 0.0
    assert math.isnan(scores["citation_recall"])


def test_false_abstention_scores_zero(corpus) -> None:
    g = corpus.canonical
    q = next(q for q in corpus.gold if q.answer_type == "entity")
    assert score_response(g, g.alias_index(), q, _resp([], "NO_ANSWER", abstained=True))["correct"] == 0.0


def test_number_answer(corpus) -> None:
    g = corpus.canonical
    q = next(q for q in corpus.gold if q.answer_type == "number")
    good = score_response(g, g.alias_index(), q, _resp([], f"They own {q.answer[0]} tier-1 services."))
    bad = score_response(g, g.alias_index(), q, _resp([str(int(q.answer[0]) + 1)]))
    assert (good["correct"], bad["correct"]) == (1.0, 0.0)
