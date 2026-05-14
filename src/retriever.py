import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, corpus_path: str):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.texts = [
            doc["title"] + ". " + doc["text"]
            for doc in self.documents
        ]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.doc_vectors).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

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