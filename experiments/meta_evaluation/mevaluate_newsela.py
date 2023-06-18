import sys

from metrics.bleu import BLEU
from metrics.sari import SARI
from metrics.fkgl_score import FKGL
from metrics.bscore import BERTScore
from metrics.lens_metric import LENS_metric
from data_loader.newsela import SimplificationNewsela
from evaluation.pearson_corr import calculate_pearson


dataset = SimplificationNewsela(sys.argv[1], sys.argv[2])
complex, simplifications, references, ratings = dataset.get_inputs_for_pearson()
for metric in [FKGL(), BLEU(), SARI(), LENS_metric(sys.argv[3]), BERTScore(type="precision")]:
    print("Pearson correlation values for", metric.name)
    pearson_vals = calculate_pearson(complex, simplifications, references, ratings, metric)
    for dimension, corr_value in zip(dataset.dimensions, pearson_vals):
        print(dimension, corr_value)
    print("*" * 30)