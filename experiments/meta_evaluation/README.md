We provide code to replicate the metric evaluation results (Table 2) in our paper.

## Installation Instructions

We recommend creating a new conda environment and installing the required packages using the following commands.

```bash
    conda create -n lens python=3.9
    conda activate lens
    pip install -r requirments.txt
```

## Run Scripts
First, download the LENS checkpoint from [here](https://drive.google.com/drive/folders/1unqQ_bpUjOdXcjTV6YmgCWF3l0sMcCvv).

To replicate results on SimpEval_2022, please run the following command.

```bash
   python3 mevaluate_simpeval.py ../../data/simpeval_2022.csv data/simpeval_2022/references <path to the lens checkpoint>
```


To replicate results on [WikiDA](https://github.com/feralvam/metaeval-simplification) released by Alva-Manchego et al. 2021, please run the following command.

```bash
   python3 mevaluate_wikida.py data/wikida/simplicity_DA.csv data/wikida/references <path to the lens checkpoint>
```

To request the Newsela-Likert ratings dataset, please first obtain access to the [Newsela corpus](https://newsela.com/data/), then contact the authors. 
Then, please run the following command to replicate the results.

```bash
   python3 mevaluate_newsela.py <ratings folder> <references folder> <path to the lens checkpoint>
```
