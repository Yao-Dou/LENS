import math

from .models import load_from_checkpoint

class LENS:
    def __init__(
            self, 
            path: str, 
            rescale: bool = False
        ):
        self.rescale = rescale
        self.model = load_from_checkpoint(path)

    def score(
            self, 
            complex: list[str], 
            simplified: list[str], 
            references: list[list[str]], 
            batch_size: int = 16, 
            devices: list[int] = None
        ):
        all_data = []
        for com, hyp, refs in zip(complex, simplified, references):
            for ref in refs:
                data = {"src": com.lower(), "mt": hyp.lower(), "ref": ref.lower()}
                all_data.append(data)

        prediction = self.model.predict(
            all_data, 
            batch_size=batch_size, 
            devices=devices
        )
        orig_scores = prediction.scores

        # Re-format scores
        ind = 0
        scores = []
        for com, simp, refs in zip(complex, simplified, references):
            fscores = []
            for _ in refs:
                score = orig_scores[ind]

                # Optionally re-scale scores
                if self.rescale:
                    score = .5 * (math.erf(orig_scores[ind] / 2 ** .5) + 1)
                
                score *= 100.0
                fscores.append(score)
                ind += 1
            scores.append(max(fscores))

        assert len(scores) == len(simplified) == len(references) == len(complex)
        return scores


class LENS_SALSA:
    def __init__(
            self, 
            path: str
        ):
        self.model = load_from_checkpoint(path)
        self.use_references = False

        self.target_column = self.model.target_column
        self.source_column = self.model.source_column

        self.is_tagging_model = self.model.word_level
        if self.is_tagging_model:
            self.continuous_word_labels = self.model.continuous_word_labels
            self.span_tokens = self.model.span_tokens
        else:
            self.continuous_word_labels = False
            self.span_tokens = None

    def recover_output(
            self, 
            tagging: dict, 
            threshold: float = 0.5
        ):
        """
        Convert tags of type (word, score) to "<tag>word word</tag> word"
        """
        num_sentences = len(tagging) if self.continuous_word_labels else len(tagging[self.span_tokens[0]])

        out = []
        for sent_idx in range(num_sentences):
            recovered_output = ""
            
            # Create a dict of token to sentence
            if self.continuous_word_labels:
                token_sents = tagging[sent_idx]
                num_tokens = len(token_sents)
            else:
                token_sents = {}
                for span_token in self.span_tokens:
                    token_sents[span_token] = tagging[span_token][sent_idx]
                num_tokens = len(token_sents[self.span_tokens[0]])

            curr = None
            for token_idx in range(num_tokens):
                if self.continuous_word_labels:
                    token = token_sents[token_idx][0]
                else:
                    token = token_sents[self.span_tokens[0]][token_idx][0]
                
                if token == '</s>':
                    recovered_output += token
                    continue

                state = None
                if self.continuous_word_labels:
                    state = round(int(token_sents[token_idx][1]))
                else:
                    for candidate_token in self.span_tokens:
                        if token_sents[candidate_token][token_idx][1] > threshold:
                            state = candidate_token

                if state != curr:
                    if curr != None:
                        recovered_output += f"</{curr}>"
                    if token[0] == "Ġ":
                        recovered_output += " "
                        token = token[1:]
                    if state != None:
                        recovered_output += f"<{state}>"
                    curr = state

                recovered_output += token.replace("Ġ", " ")
            split_outs = recovered_output.split('</s></s>')
            mt_tagged, src_tagged = split_outs
            out += [{
                'src_tagged': src_tagged,
                'mt_tagged': mt_tagged
            }]
        return out

    def score(
            self, 
            complex: list[str], 
            simplified: list[str], 
            references: list[list[str]] = None, 
            batch_size: int = 16, 
            devices: list[int] = None
        ):
        if references:
            # Reference eval
            all_data = []
            for com, hyp, refs in zip(complex, simplified, references):
                for ref in refs:
                    data = { "src": com.lower(), "mt": hyp.lower(), "ref": ref.lower() }
                    all_data.append(data)
        else:
            # Reference-less eval
            all_data = []
            for com, hyp in zip(complex, simplified):
                data = {
                    self.source_column: com.lower(),
                    self.target_column: hyp.lower(),
                }
                all_data.append(data)

        prediction = self.model.predict(
            all_data, 
            batch_size=batch_size, 
            devices=devices
        )

        prediction.scores = [s*100 for s in prediction.scores]
        
        return prediction.scores, prediction.metadata.subword_scores