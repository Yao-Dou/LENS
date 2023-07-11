# T5-3B and T5-11B Controlled Text Simplification Models

This repository hosts two pretrained models, `T5-3B` and `T5-11B`, trained on the WikiAuto dataset (Jiang et al., 2020)[^1^] using control tokens to generate text simplifications. The models have been trained using four control tokens: Character Length Ratio (NC), Character-level Levenshtein Similarity (LS), Dependency Tree Depth Ratio (DR), and Inverse Frequency Ratio (WR), following the methods outlined by Martin et al. (2020)[^2^].

The control tokens are used to control various aspects of the simplification process. As an example, during inference, the following values are used for the control tokens: `<NC_0.95> <LS_0.75> <DR_0.75> <WR_0.75>`. These are then prepended to the complex sentence, like so: `<NC_0.95> <LS_0.75> <DR_0.75> <WR_0.75> One side of the armed conflicts is composed mainly ...`.

## Available Models

1. T5-3B version: [`douy/T5-3B-Ctrl-Simplification`](https://huggingface.co/douy/T5-3B-Ctrl-Simplification)
2. T5-11B version: [`douy/T5-11B-Ctrl-Simplification`](https://huggingface.co/douy/T5-11B-Ctrl-Simplification)

## Loading the Model

To load the model in Python, you can use the following code:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

model_path = "douy/T5-11B-Ctrl-Simplification"
config = AutoConfig.from_pretrained(model_path)

model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_path)
```

## References

[^1^]: Jiang, C., Maddela, M., Lan, W., Zhong, Y., & Xu, W. (2020). Neural CRF Model for Sentence Alignment in Text Simplification. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 7943-7960). Association for Computational Linguistics. [https://aclanthology.org/2020.acl-main.709](https://aclanthology.org/2020.acl-main.709)

[^2^]: Martin, L., de la Clergerie, É., Sagot, B., & Bordes, A. (2020). Controllable Sentence Simplification. In Proceedings of the Twelfth Language Resources and Evaluation Conference (pp. 4689-4698). European Language Resources Association. [https://aclanthology.org/2020.lrec-1.577](https://aclanthology.org/2020.lrec-1.577)
