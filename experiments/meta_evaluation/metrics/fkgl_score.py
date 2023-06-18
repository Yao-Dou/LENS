import textstat
from .base_metric import BaseMetric

class FKGL(BaseMetric):

    name = "FKGL"

    def compute_metric(self, complex, simplified, references):

        all_scores = []
        for simp in simplified:
            score = textstat.flesch_kincaid_grade(simp)
            all_scores.append(score)
        return all_scores

