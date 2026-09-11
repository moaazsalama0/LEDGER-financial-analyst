
def get_gold_document_ids(gold_evidence):
    """
    Extract unique gold document IDs from gold evidence.
    """

    gold_ids = set()

    for evidence in gold_evidence:
        doc_id = evidence.get("source_doc_uid")

        if doc_id:
            gold_ids.add(str(doc_id))

    return gold_ids


def get_retrieved_document_ids(retrieved_results, k):
    """
    Extract unique document IDs from top-k retrieved results.
    """

    retrieved_ids = set()

    for result in retrieved_results[:k]:
        doc_id = result.get("document_id")

        if doc_id:
            retrieved_ids.add(str(doc_id))

    return retrieved_ids


def recall_at_k(
    retrieved_results,
    gold_evidence,
    k
):
    """
    Evidence/document Recall@K.

    For multi-document questions:
        recall = retrieved gold documents / total gold documents
    """

    gold_ids = get_gold_document_ids(
        gold_evidence
    )

    if not gold_ids:
        return 0.0

    retrieved_ids = get_retrieved_document_ids(
        retrieved_results,
        k
    )

    hits = gold_ids.intersection(
        retrieved_ids
    )

    return len(hits) / len(gold_ids)


def reciprocal_rank(
    retrieved_results,
    gold_evidence
):
    """
    Reciprocal rank of the first retrieved gold document.
    """

    gold_ids = get_gold_document_ids(
        gold_evidence
    )

    if not gold_ids:
        return 0.0

    for rank, result in enumerate(
        retrieved_results,
        start=1
    ):
        doc_id = result.get("document_id")

        if doc_id and str(doc_id) in gold_ids:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    retrieved_results,
    gold_evidence
):
    """
    Evaluate one retrieval result list.
    """

    return {
        "recall_at_1": recall_at_k(
            retrieved_results,
            gold_evidence,
            1
        ),
        "recall_at_3": recall_at_k(
            retrieved_results,
            gold_evidence,
            3
        ),
        "recall_at_5": recall_at_k(
            retrieved_results,
            gold_evidence,
            5
        ),
        "mrr": reciprocal_rank(
            retrieved_results,
            gold_evidence
        )
    }


def evaluate_multiple_questions(
    all_results,
    ground_truth_records
):
    """
    Evaluate retrieval over multiple questions.

    Parameters
    ----------
    all_results : dict
        Mapping:
            question_id -> retrieved results

    ground_truth_records : list
        Ground-truth records containing:
            question_id
            gold_evidence

    Returns
    -------
    dict
        Aggregate retrieval metrics.
    """

    per_question = []

    for record in ground_truth_records:

        question_id = record.get(
            "question_id"
        )

        if question_id not in all_results:
            continue

        retrieved_results = all_results[
            question_id
        ]

        gold_evidence = record.get(
            "gold_evidence",
            []
        )

        metrics = evaluate_retrieval(
            retrieved_results,
            gold_evidence
        )

        metrics["question_id"] = question_id

        per_question.append(metrics)

    if not per_question:
        return {
            "num_questions": 0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "per_question": []
        }

    n = len(per_question)

    aggregate = {
        "num_questions": n,

        "recall_at_1": sum(
            item["recall_at_1"]
            for item in per_question
        ) / n,

        "recall_at_3": sum(
            item["recall_at_3"]
            for item in per_question
        ) / n,

        "recall_at_5": sum(
            item["recall_at_5"]
            for item in per_question
        ) / n,

        "mrr": sum(
            item["mrr"]
            for item in per_question
        ) / n,

        "per_question": per_question
    }

    return aggregate
