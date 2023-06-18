
from .dataset import Dataset, remove_tokenization_artifacts, read_csv_file


OPERATIONS_MAPPING = {
    "Deletions": 0, "Paraphrases": 1, "Splittings": 2
}


class SimplificationDataset(Dataset):

    def _extract_annotations(self, dataset_rows, annotator_names):
        systems2id = {}
        for ind, system in enumerate(self.systems):
            systems2id[system] = ind

        self.annotations = {}
        self.operations = {}
        for system in self.systems:
            system_ind = systems2id[system]
            self.operations[system_ind] = [""] * len(self.complex_sentences)
            self.annotations[system_ind] = [""] * len(self.complex_sentences)

        for row_data in dataset_rows:
            complex = row_data['original'].lower()
            system_ind = systems2id[row_data['system']]
            complex_ind = self.complex_sentences_mapping[complex]

            ratings = [float(row_data[name]) for name in annotator_names]
            self.annotations[system_ind][complex_ind] = ratings
            operation = OPERATIONS_MAPPING[row_data["sentence_type"]]
            self.operations[system_ind][complex_ind] = operation

    def _extract_systems(self, dataset_rows):
        systems = sorted(list(set([row['system'] for row in dataset_rows])))
        self.systems = systems

    def _extract_complex_simple_sentence_mapping(self, dataset_rows):
        complex_sentences = set([row['original'].lower() for row in dataset_rows])
        complex_sentences = sorted(list(complex_sentences))
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

        for row in dataset_rows:
            complex = row['original'].lower()
            ind = self.complex_sentences_mapping[complex]
            system = row['system']
            sentence = row['generation']
            sentence = remove_tokenization_artifacts(sentence, complex).lower()
            self.systems_simplification_mapping[system]['simplifications'][ind] = sentence

    def __init__(self, ratings_file, data_folder):
        dataset_rows = read_csv_file(ratings_file)
        self._read_dataset(data_folder)
        self._extract_systems(dataset_rows)
        self._extract_complex_simple_sentence_mapping(dataset_rows)
        self._extract_system_simplification_mapping(dataset_rows)

        annotators = ["rating_1", "rating_2", "rating_3"]
        self._extract_annotations(dataset_rows, annotators)
