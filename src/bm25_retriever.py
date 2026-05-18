import json
import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, corpus_path: str):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.texts = [
            doc["title"] + ". " + doc["text"]
            for doc in self.documents
        ]

        self.tokenized_corpus = [tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 3):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "score": float(scores[idx])
            })

        return results