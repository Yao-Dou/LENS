LENS is our new learned metric for text simplification trained using our SimpEval datasets. 
We leveraged the code of [COMET metric](https://github.com/Unbabel/COMET) to implement LENS.

## Installation Instructions

Install from pypi with pip by

```bash
pip install lens
```

Install it from the source by:

```bash
git clone https://github.com/Yao-Dou/LENS.git
cd lens
pip install .
```

### Scoring within Python:
Please download the model checkpoint from [here](https://drive.google.com/drive/folders/1unqQ_bpUjOdXcjTV6YmgCWF3l0sMcCvv), which corresponds to the LENS(k=3) metric in our paper.

```python

import lens
from lens.lens_score import LENS

model_path = "<path of downloaded model checkpoint>"
# Original LENS is a real-valued number. 
# Rescaled version (rescale=True) rescales LENS between 0 and 100 for better interpretability. 
# You can also use the original version using rescale=False
metric = LENS(model_path, rescale=True)

complex = ["They are culturally akin to the coastal peoples of Papua New Guinea."]
simple = ["They are culturally similar to the people of Papua New Guinea."]
references = [
    [
        "They are culturally similar to the coastal peoples of Papua New Guinea.",
        "They are similar to the Papua New Guinea people living on the coast."
    ]
]
scores = metric.score(complex, simple, references, batch_size=8, gpus=1)
```


## Publications
If you use LENS, please cite our work! 

```angular2html
@misc{lens2022,
  author = {Maddela, Mounica and Dou, Yao and Heineman, David and Xu, Wei},
  title = {LENS: A Learnable Evaluation Metric for Text Simplification},
  publisher = {arXiv},
  year = {2022},
}
```




