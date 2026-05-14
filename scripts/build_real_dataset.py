import json
import random
from pathlib import Path
from datasets import load_dataset


random.seed(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).replace("\n", " ").split())


def build_squad_subset(n_questions=50):
    print("Loading SQuAD...")
    squad = load_dataset("rajpurkar/squad", split="validation")

    docs = []
    questions = []
    context_to_doc_id = {}

    count = 0

    for item in squad:
        if count >= n_questions:
            break

        question = normalize_text(item["question"])
        context = normalize_text(item["context"])
        title = normalize_text(item["title"])

        answers = item.get("answers", {})
        answer_texts = answers.get("text", [])

        if not question or not context or not answer_texts:
            continue

        answer = normalize_text(answer_texts[0])

        if context not in context_to_doc_id:
            doc_id = f"squad_doc_{len(context_to_doc_id) + 1}"
            context_to_doc_id[context] = doc_id

            docs.append({
                "id": doc_id,
                "title": title if title else f"SQuAD Document {len(docs) + 1}",
                "text": context,
                "source": "SQuAD"
            })

        questions.append({
            "id": f"squad_q_{count + 1}",
            "question": question,
            "answer": answer,
            "type": "one-hop",
            "source": "SQuAD",
            "supporting_doc_ids": [context_to_doc_id[context]]
        })

        count += 1

    return docs, questions


def build_hotpotqa_subset(n_questions=50):
    print("Loading HotpotQA...")
    hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    docs = []
    questions = []
    seen_doc_keys = set()

    count = 0

    for item in hotpot:
        if count >= n_questions:
            break

        question = normalize_text(item["question"])
        answer = normalize_text(item["answer"])

        if not question or not answer:
            continue

        context = item["context"]
        titles = context["title"]
        sentences_groups = context["sentences"]

        supporting_doc_ids = []

        for title, sentences in zip(titles, sentences_groups):
            text = normalize_text(" ".join(sentences))
            title = normalize_text(title)

            if not text:
                continue

            doc_key = f"{title}::{text[:80]}"

            if doc_key in seen_doc_keys:
                continue

            seen_doc_keys.add(doc_key)

            doc_id = f"hotpot_doc_{len(docs) + 1}"

            docs.append({
                "id": doc_id,
                "title": title,
                "text": text,
                "source": "HotpotQA"
            })

            supporting_doc_ids.append(doc_id)

        if len(supporting_doc_ids) < 2:
            continue

        questions.append({
            "id": f"hotpot_q_{count + 1}",
            "question": question,
            "answer": answer,
            "type": "multi-hop",
            "source": "HotpotQA",
            "supporting_doc_ids": supporting_doc_ids[:5]
        })

        count += 1

    return docs, questions


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    squad_docs, squad_questions = build_squad_subset(n_questions=50)
    hotpot_docs, hotpot_questions = build_hotpotqa_subset(n_questions=50)

    corpus = squad_docs + hotpot_docs
    test_questions = squad_questions + hotpot_questions

    random.shuffle(corpus)
    random.shuffle(test_questions)

    save_json(DATA_DIR / "corpus.json", corpus)
    save_json(DATA_DIR / "test_questions.json", test_questions)

    print("\nDone!")
    print(f"Corpus documents: {len(corpus)}")
    print(f"Test questions: {len(test_questions)}")
    print(f"One-hop questions: {len(squad_questions)}")
    print(f"Multi-hop questions: {len(hotpot_questions)}")
    print("\nSaved:")
    print(" - data/corpus.json")
    print(" - data/test_questions.json")


if __name__ == "__main__":
    main()