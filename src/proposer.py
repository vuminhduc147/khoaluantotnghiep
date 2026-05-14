import random


class QuestionProposer:
    def __init__(self):
        self.templates = [
            {
                "type": "one-hop",
                "difficulty": "easy",
                "question": "What is {concept}?",
            },
            {
                "type": "one-hop",
                "difficulty": "medium",
                "question": "What is the main purpose of {concept}?",
            },
            {
                "type": "multi-hop",
                "difficulty": "hard",
                "question": "How is {concept1} related to {concept2}?",
            },
            {
                "type": "multi-hop",
                "difficulty": "hard",
                "question": "Why can {concept1} improve {concept2}?",
            }
        ]

        self.concepts = [
            "Transformer",
            "BERT",
            "GPT",
            "RAG",
            "multi-hop question answering",
            "language agent",
            "self-evolving agent",
            "Dr. Zero"
        ]

    def generate(self, n_questions: int = 10):
        questions = []

        for i in range(n_questions):
            template = random.choice(self.templates)

            if "{concept1}" in template["question"]:
                c1, c2 = random.sample(self.concepts, 2)
                question = template["question"].format(
                    concept1=c1,
                    concept2=c2
                )
            else:
                c = random.choice(self.concepts)
                question = template["question"].format(concept=c)

            questions.append({
                "id": f"syn_{i+1}",
                "question": question,
                "type": template["type"],
                "difficulty": template["difficulty"]
            })

        return questions