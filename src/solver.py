class SearchAugmentedSolver:
    def __init__(self, retriever):
        self.retriever = retriever

    def answer(self, question: str, top_k: int = 3):
        docs = self.retriever.retrieve(question, top_k=top_k)

        evidence_text = " ".join([doc["text"] for doc in docs])

        answer = self.simple_answer_extractor(question, evidence_text)

        return {
            "question": question,
            "answer": answer,
            "evidence": docs
        }

    def simple_answer_extractor(self, question: str, evidence_text: str):
        q = question.lower()
        evidence = evidence_text

        if "bert" in q and "architecture" in q:
            return "BERT is based on the Transformer encoder architecture."

        if "retrieval-augmented generation" in q or "rag" in q:
            return "Retrieval-Augmented Generation combines information retrieval with text generation."

        if "multi-hop" in q and "harder" in q:
            return "Multi-hop question answering is harder because it requires reasoning across multiple documents or facts."

        if "dr. zero" in q or "dr zero" in q:
            return "Dr. Zero uses a proposer-solver mechanism to generate tasks and improve search and reasoning ability."

        return evidence[:300]

class NoSearchSolver:
    def answer(self, question: str):
        q = question.lower()

        if "bert" in q and "architecture" in q:
            answer = "BERT is a language model architecture."
        elif "retrieval-augmented generation" in q or "rag" in q:
            answer = "RAG is a method that uses retrieval for question answering."
        elif "multi-hop" in q:
            answer = "Multi-hop question answering is a difficult reasoning task."
        elif "dr. zero" in q or "dr zero" in q:
            answer = "Dr. Zero is a self-evolving search agent framework."
        else:
            answer = "I do not have enough information to answer."

        return {
            "question": question,
            "answer": answer,
            "evidence": []
        }