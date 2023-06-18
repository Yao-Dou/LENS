import lens
from lens.lens_score import LENS
from .base_metric import BaseMetric


class LENS_metric(BaseMetric):

    name = "LENS"

    def __init__(self, model_path):
        self.lens_metric = LENS(model_path, rescale=True)
        super().__init__()

    def compute_metric(self, complex, simplified, references):
        scores = self.lens_metric.score(complex, simplified, references,  batch_size=8, gpus=1)
        return scores