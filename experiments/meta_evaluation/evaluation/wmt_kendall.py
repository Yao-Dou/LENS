import numpy as np
from enum import Enum
from collections import Counter
from data_loader.simplification import OPERATIONS_MAPPING


class PairType(Enum):
	CONCORDANT = "CONCORDANT",
	DISCORDANT = "DISCORDANT",
	SKIPPED = "SKIPPED"


def combine_human_scores(h_score1, h_score2, threshold):
	assert len(h_score1) == len(h_score2)
	h_diff = []
	for h1, h2 in zip(h_score1, h_score2):
		if abs(h1 - h2) <= threshold:
			h_diff.append(0)
		else:
			h_diff.append(np.sign(h1 - h2))
	return h_diff


def categorize_pair(h_diff, m_diff):
	total_anns = len(h_diff)
	h_diff = Counter(h_diff)
	h_diff = [k for k, v in h_diff.items() if v > int(total_anns / 2)]

	if len(h_diff) > 0 and h_diff[0]:
		assert len(h_diff) == 1 and h_diff[0] != 0
		if m_diff * h_diff[0] > 0:
			return PairType.CONCORDANT
		else:
			return PairType.DISCORDANT
	return PairType.SKIPPED


def calculate_wmt_kendall_tau(metric_scores, dataset, all_flag=False, threshold=5):

	n_sys = len(dataset.systems)
	n_cols = len(dataset.complex_sentences)

	operations = dataset.operations
	annotations = dataset.annotations

	num_operations = len(OPERATIONS_MAPPING)
	pair_categorization = {
		PairType.CONCORDANT: [0] * num_operations,
		PairType.DISCORDANT: [0] * num_operations,
		PairType.SKIPPED: [0] * num_operations
	}

	for col in range(n_cols):
		for sys1 in range(n_sys):
			for sys2 in range(sys1 + 1, n_sys):
				if operations[sys1][col] == operations[sys2][col]:
					op = int(operations[sys1][col])
					op = 0 if all_flag else op
					
					m_score1 = metric_scores[sys1][col]
					m_score2 = metric_scores[sys2][col]
					m_diff = m_score1 - m_score2

					h_score1 = annotations[sys1][col]
					h_score2 = annotations[sys2][col]
					h_diff = combine_human_scores(h_score1, h_score2, threshold)
					category = categorize_pair(h_diff, m_diff)
					pair_categorization[category][op] += 1

	taus = []
	for i in range(num_operations):
		concordant = pair_categorization[PairType.CONCORDANT][i]
		discordant = pair_categorization[PairType.DISCORDANT][i]
		tau = 0
		if (concordant + discordant) > 0:
			tau = (concordant - discordant) / (concordant + discordant)
		taus.append(tau)
	return taus
