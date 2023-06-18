import csv
import numpy as np

from .dataset import Dataset, remove_tokenization_artifacts, read_csv_file


class SimplicityDA(Dataset):

    def _extract_annotations(self, dataset_rows):

        systems2id = {system: ind for ind, system in enumerate(self.systems)}

        self.grammar = {}
        self.meaning = {}
        self.simplicity = {}
        for system in self.systems:
            system_ind = systems2id[system]
            self.grammar[system_ind] = [-10000] * len(self.complex_sentences)
            self.meaning[system_ind] = [-10000] * len(self.complex_sentences)
            self.simplicity[system_ind] = [-10000] * len(self.complex_sentences)

        for row_data in dataset_rows:
            complex = row_data['orig_sent'].lower()
            system_ind = systems2id[row_data['sys_name']]
            complex_ind = self.complex_sentences_mapping[complex]
            self.grammar[system_ind][complex_ind] = float(row_data["fluency_zscore"])
            self.meaning[system_ind][complex_ind] = float(row_data["meaning_zscore"])
            self.simplicity[system_ind][complex_ind] = float(row_data["simplicity_zscore"])


    def _extract_systems(self, dataset_rows):
        systems = sorted(list(set([row['sys_name'] for row in dataset_rows])))
        self.systems = systems


    def _extract_complex_simple_sentence_mapping(self, dataset_rows):
        complex_sentences = set([row['orig_sent'].lower() for row in dataset_rows])
        complex_sentences = sorted(list(complex_sentences))
        complex_sentences_mapping = {}
        for id, complex in enumerate(complex_sentences):
            complex_sentences_mapping[complex] = id
        self.complex_sentences = complex_sentences
        self.complex_sentences_mapping = complex_sentences_mapping


    def _extract_system_simplification_mapping(self, dataset_rows):
        systems2id = {system: ind  for ind, system in enumerate(self.systems)}

        num_complex = len(self.complex_sentences)
        self.systems_simplification_mapping = {}
        for system in self.systems:
            self.systems_simplification_mapping[system] = {}
            self.systems_simplification_mapping[system]['id'] = systems2id[system]
            self.systems_simplification_mapping[system]['simplifications'] = [""] * num_complex

        for row in dataset_rows:
            complex = row['orig_sent'].lower()
            ind = self.complex_sentences_mapping[complex]
            system = row['sys_name']
            sentence = row['simp_sent'].lower()
            self.systems_simplification_mapping[system]['simplifications'][ind] = sentence

    def __init__(self, ratings_file, data_folder):
        dataset_rows = read_csv_file(ratings_file)
        self._read_dataset(data_folder)
        self._extract_systems(dataset_rows)
        self._extract_complex_simple_sentence_mapping(dataset_rows)
        self._extract_system_simplification_mapping(dataset_rows)
        self._extract_annotations(dataset_rows)
