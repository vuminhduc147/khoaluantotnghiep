import json
import pandas as pd
from tqdm import tqdm

from retriever import TfidfRetriever
from solver import SearchAugmentedSolver, NoSearchSolver
from proposer import QuestionProposer
from evaluator import SimpleEvaluator
from curriculum import DifficultyBasedFilter


def load_test_questions():
    with open("data/test_questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def run_no_search_baseline():
    print("\n=== Running No Search Baseline ===")

    solver = NoSearchSolver()
    evaluator = SimpleEvaluator()
    test_questions = load_test_questions()

    results = []

    for item in tqdm(test_questions):
        output = solver.answer(item["question"])

        eval_result = evaluator.evaluate_with_gold(
            output["answer"],
            item["answer"],
            output["evidence"]
        )

        results.append({
            "method": "no_search",
            "id": item["id"],
            "question": item["question"],
            "type": item.get("type", "unknown"),
            "source": item.get("source", "unknown"),
            "gold_answer": item["answer"],
            "predicted_answer": output["answer"],
            "top_evidence": None,
            "evidence_score": 0,
            "exact_match": eval_result["exact_match"],
            "retrieval_hit": eval_result["retrieval_hit"],
            "score": eval_result["score"]
        })

    df = pd.DataFrame(results)
    df.to_csv("results/no_search_results.csv", index=False, encoding="utf-8-sig")

    print(df.head())
    print("\nAverage score:", df["score"].mean())
    return df


def run_baseline_search():
    print("\n=== Running Search Agent Baseline ===")

    retriever = TfidfRetriever("data/corpus.json")
    solver = SearchAugmentedSolver(retriever)
    evaluator = SimpleEvaluator()
    test_questions = load_test_questions()

    results = []

    for item in tqdm(test_questions):
        output = solver.answer(item["question"])

        eval_result = evaluator.evaluate_with_gold(
            output["answer"],
            item["answer"],
            output["evidence"]
        )

        results.append({
            "method": "search_agent",
            "id": item["id"],
            "question": item["question"],
            "type": item.get("type", "unknown"),
            "source": item.get("source", "unknown"),
            "gold_answer": item["answer"],
            "predicted_answer": output["answer"],
            "top_evidence": output["evidence"][0]["title"] if output["evidence"] else None,
            "evidence_score": output["evidence"][0]["score"] if output["evidence"] else 0,
            "exact_match": eval_result["exact_match"],
            "retrieval_hit": eval_result["retrieval_hit"],
            "score": eval_result["score"]
        })

    df = pd.DataFrame(results)
    df.to_csv("results/baseline_search_results.csv", index=False, encoding="utf-8-sig")

    print(df.head())
    print("\nAverage score:", df["score"].mean())
    return df


def run_self_evolving_agent():
    print("\n=== Running Mini Self-Evolving Search Agent ===")

    proposer = QuestionProposer()
    curriculum = DifficultyBasedFilter(keep_difficulties=["medium", "hard"])

    retriever = TfidfRetriever("data/corpus.json")
    solver = SearchAugmentedSolver(retriever)
    evaluator = SimpleEvaluator()

    all_results = []

    n_rounds = 3

    for round_id in range(1, n_rounds + 1):
        print(f"\n--- Self-evolving round {round_id} ---")

        synthetic_questions = proposer.generate(n_questions=30)
        selected_questions = curriculum.filter(synthetic_questions)

        for q in tqdm(selected_questions):
            output = solver.answer(q["question"])

            eval_result = evaluator.evaluate_without_gold(
                output["answer"],
                output["evidence"]
            )

            all_results.append({
                "method": "self_evolving_search_agent",
                "round": round_id,
                "id": q["id"],
                "question": q["question"],
                "type": q["type"],
                "difficulty": q["difficulty"],
                "predicted_answer": output["answer"],
                "top_evidence": output["evidence"][0]["title"] if output["evidence"] else None,
                "evidence_score": output["evidence"][0]["score"] if output["evidence"] else 0,
                "auto_score": eval_result["score"]
            })

    df = pd.DataFrame(all_results)
    df.to_csv("results/self_evolving_results.csv", index=False, encoding="utf-8-sig")

    print(df.head())
    print("\nAverage auto score:", df["auto_score"].mean())
    return df

def run_proposed_on_real_test():
    print("\n=== Running Proposed Self-Evolving Search Agent on Real Test Set ===")

    proposer = QuestionProposer()
    curriculum = DifficultyBasedFilter(keep_difficulties=["medium", "hard"])

    retriever = TfidfRetriever("data/corpus.json")
    solver = SearchAugmentedSolver(retriever)
    evaluator = SimpleEvaluator()
    test_questions = load_test_questions()

    synthetic_questions = proposer.generate(n_questions=100)
    selected_questions = curriculum.filter(synthetic_questions)

    expansion_terms = []
    for q in selected_questions:
        words = q["question"].lower().split()
        for w in words:
            if len(w) > 5:
                expansion_terms.append(w)

    expansion_terms = list(set(expansion_terms))[:20]
    expansion_text = " ".join(expansion_terms)

    results = []

    for item in tqdm(test_questions):
        expanded_question = item["question"] + " " + expansion_text

        docs = retriever.retrieve(expanded_question, top_k=5)

        evidence_text = " ".join([doc["text"] for doc in docs])
        answer = solver.simple_answer_extractor(item["question"], evidence_text)

        output = {
            "question": item["question"],
            "answer": answer,
            "evidence": docs
        }

        eval_result = evaluator.evaluate_with_gold(
            output["answer"],
            item["answer"],
            output["evidence"]
        )

        results.append({
            "method": "proposed_self_evolving_search_agent",
            "id": item["id"],
            "question": item["question"],
            "type": item.get("type", "unknown"),
            "source": item.get("source", "unknown"),
            "gold_answer": item["answer"],
            "predicted_answer": output["answer"],
            "top_evidence": output["evidence"][0]["title"] if output["evidence"] else None,
            "evidence_score": output["evidence"][0]["score"] if output["evidence"] else 0,
            "exact_match": eval_result["exact_match"],
            "retrieval_hit": eval_result["retrieval_hit"],
            "score": eval_result["score"]
        })

    df = pd.DataFrame(results)
    df.to_csv("results/proposed_real_test_results.csv", index=False, encoding="utf-8-sig")

    print(df.head())
    print("\nAverage score:", df["score"].mean())
    return df

def summarize_results(no_search_df, search_df, self_evolving_df, proposed_df=None):
    print("\n=== Summary ===")

    summary_rows = []

    summary_rows.append({
        "method": "No Search Agent",
        "exact_match": no_search_df["exact_match"].mean(),
        "retrieval_hit": no_search_df["retrieval_hit"].mean(),
        "avg_score": no_search_df["score"].mean()
    })

    summary_rows.append({
        "method": "Search Agent",
        "exact_match": search_df["exact_match"].mean(),
        "retrieval_hit": search_df["retrieval_hit"].mean(),
        "avg_score": search_df["score"].mean()
    })

    if proposed_df is not None:
        summary_rows.append({
            "method": "Proposed Self-Evolving Search Agent",
            "exact_match": proposed_df["exact_match"].mean(),
            "retrieval_hit": proposed_df["retrieval_hit"].mean(),
            "avg_score": proposed_df["score"].mean()
        })

    synthetic_summary = pd.DataFrame([{
        "method": "Self-Evolving Synthetic Evaluation",
        "auto_score": self_evolving_df["auto_score"].mean()
    }])
    synthetic_summary.to_csv("results/synthetic_summary_results.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/summary_results.csv", index=False, encoding="utf-8-sig")

    print(summary_df)

    type_summary = search_df.groupby("type")[["exact_match", "retrieval_hit", "score"]].mean().reset_index()
    type_summary.to_csv("results/search_by_question_type.csv", index=False, encoding="utf-8-sig")

    print("\nSearch Agent by question type:")
    print(type_summary)

    print("\nSynthetic self-evolving summary:")
    print(synthetic_summary)

def save_type_comparison(no_search_df, search_df, proposed_df):
    all_df = pd.concat([no_search_df, search_df, proposed_df], ignore_index=True)

    type_summary = (
        all_df
        .groupby(["method", "type"])[["exact_match", "retrieval_hit", "score"]]
        .mean()
        .reset_index()
    )

    type_summary.to_csv(
        "results/type_comparison_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== Type Comparison ===")
    print(type_summary)


if __name__ == "__main__":
    no_search_df = run_no_search_baseline()
    search_df = run_baseline_search()
    self_evolving_df = run_self_evolving_agent()
    proposed_df = run_proposed_on_real_test()

    summarize_results(no_search_df, search_df, self_evolving_df, proposed_df)
    save_type_comparison(no_search_df, search_df, proposed_df)


