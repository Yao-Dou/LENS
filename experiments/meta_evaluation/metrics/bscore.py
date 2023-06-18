import bert_score
from .base_metric import BaseMetric


class BERTScore(BaseMetric):

	def __init__(self, type="F1", self_flag=False):
		self.name = "BERTScore-" + type
		self.type = type
		self.self_flag = self_flag
		super().__init__()

	def compute_metric(self, complex, simplified, references):

		all_comps, all_refs, all_cands = [], [], []
		for com, hyp, refs in zip(complex, simplified, references):
			for ref in refs:
				all_refs.append(ref.lower())
				all_cands.append(hyp.lower())
				all_comps.append(com.lower())

		if self.self_flag:
			(P, R, F), _ = bert_score.score(all_cands, all_comps, lang="en", return_hash=True, verbose=True, idf=False,
											model_type="roberta-large")
		else:
			(P, R, F), _ = bert_score.score(all_cands, all_refs, lang="en", return_hash=True, verbose=True, idf=False,
										model_type="roberta-large")
		if self.type == "recall":
			all_scores = R
		elif self.type == "precision":
			all_scores = P
		else:
			all_scores = F

		ind = 0
		scores = []
		for com, simp, refs in zip(complex, simplified, references):
			fscores = []
			for _ in refs:
				fscores.append(all_scores[ind].item())
				ind += 1
			scores.append(max(fscores))

		assert len(scores) == len(simplified) == len(references) == len(complex)
		return scores

