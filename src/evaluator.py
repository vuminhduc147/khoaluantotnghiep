class SimpleEvaluator:
    def exact_match(self, predicted: str, gold: str):
        pred = predicted.lower().strip()
        gold = gold.lower().strip()

        return int(gold in pred or pred in gold)

    def retrieval_hit(self, gold: str, evidence: list):
        gold = gold.lower().strip()

        if not evidence:
            return 0

        for doc in evidence:
            text = doc.get("text", "").lower()
            if gold and gold in text:
                return 1

        return 0

    def evaluate_with_gold(self, predicted: str, gold: str, evidence=None):
        em = self.exact_match(predicted, gold)

        if evidence is None:
            hit = 0
        else:
            hit = self.retrieval_hit(gold, evidence)

        score = 0.5 * em + 0.5 * hit

        return {
            "exact_match": em,
            "retrieval_hit": hit,
            "score": score
        }

    def evaluate_without_gold(self, answer: str, evidence: list):
        answer_len = len(answer.split())
        evidence_score = 1 if len(evidence) > 0 and evidence[0]["score"] > 0 else 0

        if answer_len < 5:
            quality_score = 0
        elif answer_len <= 100:
            quality_score = 1
        else:
            quality_score = 0.5

        final_score = 0.5 * evidence_score + 0.5 * quality_score

        return {
            "evidence_score": evidence_score,
            "answer_quality": quality_score,
            "score": final_score
        }