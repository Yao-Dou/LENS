import sys
import csv
import glob
import numpy as np
from collections import Counter


from .dataset import Dataset, read_csv_file

SYSTEMS = ['Input.SOURCE_SENTENCE', 'Input.REF_SENTENCE', 'Input.EDITNTS', 'Input.TRANSFORMER',
           'Input.OUR_MODEL', 'Input.HYBRID']


def clean(source):
    return source.replace(" .", ".").replace(" ,", ",").replace(" 's", "'s").replace("( ", "(").replace(" )", ")")


def group_ratings(dataset_rows):
    hits = {}
    for i, row_data in enumerate(dataset_rows):
        hitid = row_data["HITId"]
        hits.setdefault(hitid, [])

        grammar = [int(row_data['Answer.grammar_0_' + str(i)]) for i in range(5)]
        meaning = [int(row_data['Answer.meaning_0_' + str(i)]) for i in range(5)]
        simplicity = [int(row_data['Answer.simplicity_0_' + str(i)]) for i in range(5)]
        simps = [row_data[s].lower() for s in SYSTEMS]
        ratings = {
                "grammar": np.array(grammar),
                "meaning": np.array(meaning),
                "simplicity": np.array(simplicity),
                "source": clean(simps[0]),
                "simplifications": simps[1:]
        }
        hits[hitid].append(ratings)
    return hits


class SimplificationNewsela(Dataset):

    def __init__(self, mturk_folder, data_folder):
        dataset_rows = []
        for mturk_file in glob.glob(mturk_folder + "/*"):
            tmp_rows = read_csv_file(mturk_file)
            dataset_rows.extend(tmp_rows)
        dataset_rows = group_ratings(dataset_rows)

        self._extract_systems()
        self._read_dataset(data_folder)
        self._extract_complex_simple_sentence_mapping(dataset_rows)
        self._extract_system_simplification_mapping(dataset_rows)
        self._extract_annotations(dataset_rows)

    def _extract_systems(self):
        self.systems = SYSTEMS[1:]

    def _extract_complex_simple_sentence_mapping(self, dataset_rows):
        complex_sentences = []
        for hit, hit_data in dataset_rows.items():
            assert len(set([data["source"] for data in hit_data])) == 1
            complex_sentences.append(hit_data[0]["source"])

        complex_sentences_mapping = {}
        for id, complex in enumerate(complex_sentences):
            complex_sentences_mapping[complex] = id
        self.complex_sentences = complex_sentences
        self.complex_sentences_mapping = complex_sentences_mapping

    def _extract_system_simplification_mapping(self, dataset_rows):
        systems2id = {}
        for ind, system in enumerate(self.systems):
            systems2id[system] = ind

        num_complex = len(self.complex_sentences)
        self.systems_simplification_mapping = {}
        for system in self.systems:
            self.systems_simplification_mapping[system] = {}
            self.systems_simplification_mapping[system]['id'] = systems2id[system]
            self.systems_simplification_mapping[system]['simplifications'] = [""] * num_complex

        for hit, hit_data in dataset_rows.items():
            assert len(set([tuple(hit["simplifications"]) for hit in hit_data])) == 1
            complex = hit_data[0]['source'].lower()
            for sind, sentence in enumerate(hit_data[0]["simplifications"]):
                ind = self.complex_sentences_mapping[complex]
                system = self.systems[sind]
                self.systems_simplification_mapping[system]['simplifications'][ind] = sentence.lower()

    def _extract_annotations(self, dataset_rows):
        systems2id = {}
        for ind, system in enumerate(self.systems):
            systems2id[system] = ind

        self.dimensions = ["grammar", "meaning", "simplicity"]
        self.operations = {}
        self.annotations = {dim : {} for dim in self.dimensions}
        for system in self.systems:
            system_ind = systems2id[system]
            self.operations[system_ind] = [0] * len(self.complex_sentences)
            for dim in self.dimensions:
                self.annotations[dim][system_ind] = {}

        for hit, hit_data in dataset_rows.items():
            assert len(set([tuple(hit["simplifications"]) for hit in hit_data])) == 1
            complex = hit_data[0]['source'].lower()
            for data in hit_data:
                cind = self.complex_sentences_mapping[complex]
                for dim in self.dimensions:
                    for sind, rating in enumerate(data[dim]):
                        self.annotations[dim][sind].setdefault(cind, [])
                        self.annotations[dim][sind][cind].append(rating.item())

    def get_inputs_for_pearson(self):
        done = set()
        grammar, meaning, simplicity = [], [], []
        all_complex, all_simp, all_refs = [], [], []
        for cid, complex in enumerate(self.complex_sentences):
            for sid, system in enumerate(self.systems):
                simplification = self.systems_simplification_mapping[system]['simplifications'][cid]
                if (complex, simplification) not in done:
                    all_complex.append(complex)
                    all_simp.append(simplification)
                    all_refs.append(self.references[complex])
                    rating = np.mean(self.annotations["grammar"][sid][cid])
                    grammar.append(rating)
                    rating = np.mean(self.annotations["meaning"][sid][cid])
                    meaning.append(rating)
                    rating = np.mean(self.annotations["simplicity"][sid][cid])
                    simplicity.append(rating)
                    done.add((complex, simplification))
        return all_complex, all_simp, all_refs, [grammar, meaning, simplicity]

