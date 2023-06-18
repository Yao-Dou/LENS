import sys

from metrics.bleu import BLEU
from metrics.sari import SARI
from metrics.fkgl_score import FKGL
from metrics.bscore import BERTScore
from metrics.lens_metric import LENS_metric
from data_loader.simplicity_da import SimplicityDA
from evaluation.pearson_corr import calculate_pearson


def get_input_for_metrics(dataset):

    # List of all the complex sentences from SimpEval_PAST then need to removed from the WikiDA dataset.
    sentences = [line.strip().lower() for line in open("data/complex_sents.txt")]

    fluency, meaning, simplicity = [], [], []
    all_complex, all_simp, all_refs = [], [], []
    for cid, complex in enumerate(dataset.complex_sentences):
        for sid, system in enumerate(dataset.systems):
            simplification = dataset.systems_simplification_mapping[system]['simplifications'][cid]
            if dataset.grammar[sid][cid] != -10000 and complex not in sentences:
                all_complex.append(complex)
                all_simp.append(simplification)
                all_refs.append(dataset.references[complex])
                fluency.append(dataset.grammar[sid][cid])
                meaning.append(dataset.meaning[sid][cid])
                simplicity.append(dataset.simplicity[sid][cid])

    return all_complex, all_simp, all_refs, [fluency, meaning, simplicity]


dataset = SimplicityDA(sys.argv[1], sys.argv[2])
complex, simplifications, references, ratings = get_input_for_metrics(dataset)
for metric in [FKGL(), BLEU(), SARI(), LENS_metric(sys.argv[3]), BERTScore(type="precision")]:
    print("Pearson correlation values for", metric.name)
    pearson_vals = calculate_pearson(complex, simplifications, references, ratings, metric)
    print(pearson_vals)
    print("*" * 30)

