import math

from .models import load_from_checkpoint

class LENS:

    def __init__(self, path, rescale=False):
        self.rescale = rescale
        self.model = load_from_checkpoint(path)


    def score(self, complex, simplified, references,
              batch_size=16, gpus=1):

        all_data = []
        for com, hyp, refs in zip(complex, simplified, references):
            for ref in refs:
                data = {"src": com.lower(), "mt": hyp.lower(), "ref": ref.lower()}
                all_data.append(data)

        all_scores, _ = self.model.predict(all_data, batch_size=batch_size, gpus=gpus)

        ind = 0
        scores = []
        for com, simp, refs in zip(complex, simplified, references):
            fscores = []
            for _ in refs:
                score = all_scores[ind]
                if self.rescale:
                    score = .5 * (math.erf(all_scores[ind] / 2 ** .5) + 1)
                    score = score * 100.0
                fscores.append(score)
                ind += 1
            scores.append(max(fscores))

        assert len(scores) == len(simplified) == len(references) == len(complex)
        return scores
