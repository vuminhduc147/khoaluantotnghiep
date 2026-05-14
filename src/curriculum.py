class DifficultyBasedFilter:
    def __init__(self, keep_difficulties=None):
        if keep_difficulties is None:
            keep_difficulties = ["medium", "hard"]

        self.keep_difficulties = keep_difficulties

    def filter(self, questions):
        selected = []

        for q in questions:
            if q.get("difficulty") in self.keep_difficulties:
                selected.append(q)

        return selected