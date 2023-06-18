import csv
import numpy as np
from string import punctuation


def remove_tokenization_artifacts(simp, orginal):
	simp = simp.replace(" || ", " ").replace("||", "").replace(" <SEP> ", " ")

	simp_tokens = simp.split()
	simp_new = simp
	for i, token in enumerate(simp_tokens):
		if 0 < i < len(simp_tokens) - 1 and token in punctuation:
			substrboth = simp_tokens[i - 1] + token + simp_tokens[i + 1]
			substrleft = simp_tokens[i - 1] + token
			substright = token + simp_tokens[i + 1]
			if substrboth in orginal:
				new_str = simp_tokens[i - 1] + " " + token + " " + simp_tokens[i + 1]
				simp_new = simp_new.replace(new_str, substrboth)
			elif substrleft in orginal:
				new_str = simp_tokens[i - 1] + " " + token
				simp_new = simp_new.replace(new_str, substrleft)
			elif substright in orginal:
				new_str = token + " " + simp_tokens[i + 1]
				simp_new = simp_new.replace(new_str, substright)
	return simp_new


def read_csv_file(ratings_file, delimiter=","):
	dataset_rows = []
	with open(ratings_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=delimiter)
		fields = next(csv_reader)
		for row in csv_reader:
			row_data = {}
			for i, val in enumerate(row):
				row_data[fields[i]] = val
			dataset_rows.append(row_data)
	return dataset_rows


class Dataset:
	def _read_dataset(self, dataset_dir, lower=True):
		complex_file = dataset_dir + "/test.src"
		simple_file = dataset_dir + "/test.dst"

		complexfp = open(complex_file)
		simplefp = open(simple_file)
		complex_simple_mapping = {}
		for complex_sent, simple_refs in zip(complexfp, simplefp):
			complex_sent = complex_sent.strip()
			simple_refs = simple_refs.strip().split("\t")
			simple_refs = [sent.strip() for sent in simple_refs]
			if lower:
				complex_sent = complex_sent.lower()
				simple_refs = [sent.lower() for sent in simple_refs]
			complex_simple_mapping[complex_sent] = simple_refs
		complexfp.close()
		simplefp.close()

		self.references = complex_simple_mapping
