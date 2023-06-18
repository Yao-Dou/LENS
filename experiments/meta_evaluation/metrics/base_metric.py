import numpy as np


class BaseMetric:

    def compute_metric(self, complex, simplified, references):
        pass

    def compute_metric_dataset(self, dataset):

        all_ids = []
        complex, simplified, references = [], [], []
        for cid, org in enumerate(dataset.complex_sentences):
            for sid, system in enumerate(dataset.systems):
                simplification = dataset.systems_simplification_mapping[system]['simplifications'][cid]
                complex.append(org)
                simplified.append(simplification)
                references.append(dataset.references[org])
                all_ids.append((cid, sid))

        scores = self.compute_metric(complex, simplified, references)

        final_scores = np.zeros((len(dataset.systems), len(dataset.complex_sentences)))
        for i in range(len(all_ids)):
            cid, sid = all_ids[i]
            final_scores[sid][cid] = scores[i]
        return final_scores
