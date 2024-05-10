LENS is our new learned metric for text simplification trained using our SimpEval datasets. 
We leveraged the code of [COMET metric](https://github.com/Unbabel/COMET) to implement LENS.

**For example usage, please see our [quick start Google Collab notebook](https://colab.research.google.com/drive/1rIYrbl5xzL5b5sGUQ6zFBfwlkyIDg12O?usp=sharing)!**

## Setup

To install from [PyPi](https://pypi.org/project/lens-metric) using `pip`:

```bash
pip install lens-metric
```

To directly install from the source:

```bash
git clone https://github.com/Yao-Dou/LENS.git
cd LENS/lens
pip install .
```

### Scoring within Python

The trained model checkpoint is available on HuggingFace at [davidheineman/lens](https://huggingface.co/davidheineman/lens), which corresponds to the LENS (k=3) metric in our paper.

```python
from lens import download_model, LENS

lens_path = download_model("davidheineman/lens")

# Original LENS is a real-valued number. 
# Rescaled version (rescale=True) rescales LENS between 0 and 100 for better interpretability. 
# You can also use the original version using rescale=False
lens = LENS(lens_path, rescale=True)

complex = [
    "They are culturally akin to the coastal peoples of Papua New Guinea."
]
simple = [
    "They are culturally similar to the people of Papua New Guinea."
]
references = [[
    "They are culturally similar to the coastal peoples of Papua New Guinea.",
    "They are similar to the Papua New Guinea people living on the coast."
]]

scores = lens.score(complex, simple, references, batch_size=8, devices=[0])
print(scores) # [78.6344531130125]
```

### Reference-free Evaluation with LENS SALSA

This repo also implements the [LENS-SALSA metric](https://aclanthology.org/2023.emnlp-main.211.pdf#page=8.48), which is a *reference-free* metric trained on both word- and senetence-level quality. For more information, please see the [SALSA repository](https://github.com/davidheineman/salsa).

```python
from lens import download_model, LENS_SALSA

lens_salsa_path = download_model("davidheineman/lens-salsa")
lens_salsa = LENS_SALSA(lens_salsa_path)

complex = [
    "They are culturally akin to the coastal peoples of Papua New Guinea."
]
simple = [
    "They are culturally similar to the people of Papua New Guinea."
]

scores, word_level_scores = lens_salsa.score(complex, simple, batch_size=8, devices=[0])
print(scores) # [72.40909337997437]

# LENS-SALSA also returns an error-identification tagging, recover_output() will return the tagged output
tagged_output = lens_salsa.recover_output(word_level_scores, threshold=0.5)
print(tagged_output)
```

## Cite this Work
If you use LENS, please cite [our work](https://aclanthology.org/2023.acl-long.905)! 

```tex
@inproceedings{maddela-etal-2023-lens,
  title = "{LENS}: A Learnable Evaluation Metric for Text Simplification",
  author = "Maddela, Mounica  and
    Dou, Yao  and
    Heineman, David  and
    Xu, Wei",
  booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2023",
  address = "Toronto, Canada",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.acl-long.905",
  doi = "10.18653/v1/2023.acl-long.905",
  pages = "16383--16408",
}
```




