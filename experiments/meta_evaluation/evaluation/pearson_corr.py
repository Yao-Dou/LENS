from scipy.stats import pearsonr


def calculate_pearson(complex, simplifications, references, human_ratings, metric):
    assert len(complex) == len(simplifications) == len(references) == len(human_ratings[0])
    scores = metric.compute_metric(complex, simplifications, references)
    corr_values = [pearsonr(scores, dimen_ratings) for dimen_ratings in human_ratings]
    return corr_values
