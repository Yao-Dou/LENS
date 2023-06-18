import sys

import lens
from metrics.bleu import BLEU
from metrics.sari import SARI
from metrics.fkgl_score import FKGL
from metrics.bscore import BERTScore
from metrics.lens_metric import LENS_metric
from evaluation.wmt_kendall import calculate_wmt_kendall_tau
from data_loader.simplification import SimplificationDataset, OPERATIONS_MAPPING


dataset = SimplificationDataset(sys.argv[1], sys.argv[2])
metrics = [FKGL(), BLEU(), SARI(), LENS_metric(sys.argv[3]), BERTScore(type="precision")]
for metric in metrics:
	metric_scores = metric.compute_metric_dataset(dataset)
	taus = calculate_wmt_kendall_tau(metric_scores, dataset, all_flag=False, threshold=5)
	all_taus = calculate_wmt_kendall_tau(metric_scores, dataset, all_flag=True, threshold=5)
	print("Kendall Tau-like correlation values for ", metric.name)
	for operation in OPERATIONS_MAPPING:
		print(operation, taus[OPERATIONS_MAPPING[operation]])
	print("All", all_taus[0])
	print("*" * 30)




