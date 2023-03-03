# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
RegressionMetric
========================
    Regression Metric that learns to predict a quality assessment by looking
    at source, translation and reference.
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from lens.models.base import CometModel
from lens.models.metrics import RegressionMetrics
from lens.modules import FeedForward
from transformers.optimization import Adafactor


class RegressionMetricMultiReference(CometModel):
    """RegressionMetricMultiReference:

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the encoder frozen during training.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param hidden_sizes: Hidden sizes for the Feed Forward regression.
    :param activations: Feed Forward activation function.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-base",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[str] = None,
        validation_data: Optional[str] = None,
        hidden_sizes: List[int] = [2304, 768],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        load_weights_from_checkpoint: Optional[str] = None,
        topk: int = 1,
    ) -> None:
        super().__init__(
            nr_frozen_epochs,
            keep_embeddings_frozen,
            optimizer,
            encoder_learning_rate,
            learning_rate,
            layerwise_decay,
            encoder_model,
            pretrained_model,
            pool,
            layer,
            dropout,
            batch_size,
            train_data,
            validation_data,
            load_weights_from_checkpoint,
            "regression_metric_multi_ref",
        )
        self.save_hyperparameters()

        self.topk = topk
        print(self.topk)

        self.estimator = FeedForward(
            in_dim=self.encoder.output_units * 7,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )

    def init_metrics(self):
        self.train_metrics = RegressionMetrics(prefix="train")
        self.val_metrics = RegressionMetrics(prefix="val")
    
    def is_referenceless(self) -> bool:
        return False

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Sets the optimizers to be used during training."""
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        top_layers_parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]
        if self.layerwise_attention:
            layerwise_attn_params = [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]
            params = layer_parameters + top_layers_parameters + layerwise_attn_params
        else:
            params = layer_parameters + top_layers_parameters

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)
        # scheduler = self._build_scheduler(optimizer)
        return [optimizer], []

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """

        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        if inference:
            src_inputs = self.encoder.prepare_sample(sample["src"])
            mt_inputs = self.encoder.prepare_sample(sample["mt"])
            ref_inputs = self.encoder.prepare_sample(sample["ref"])

            src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
            mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
            ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
            inputs = {**src_inputs, **mt_inputs, **ref_inputs}
            return inputs

        all_inputs = []
        for src, mt, refs in zip(sample['src'], sample['mt'], sample['ref']):
            refs = refs.split("\t")
            num_refs = len(refs)
            src = [src] * num_refs
            mt = [mt] * num_refs

            src_inputs = self.encoder.prepare_sample(src)
            mt_inputs = self.encoder.prepare_sample(mt)
            ref_inputs = self.encoder.prepare_sample(refs)

            src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
            mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
            ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}

            inputs = {**src_inputs, **mt_inputs, **ref_inputs}
            all_inputs.append(inputs)

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return all_inputs, targets

    def estimate(
        self,
        src_sentemb: torch.Tensor,
        mt_sentemb: torch.Tensor,
        ref_sentemb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        embedded_sequences = torch.cat(
            (src_sentemb, mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
            dim=1,
        )
        return {"score": self.estimator(embedded_sequences)}

    def forward(
        self,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt_input_ids: torch.tensor,
        mt_attention_mask: torch.tensor,
        ref_input_ids: torch.tensor,
        ref_attention_mask: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        mt_sentemb = self.get_sentence_embedding(mt_input_ids, mt_attention_mask)
        ref_sentemb = self.get_sentence_embedding(ref_input_ids, ref_attention_mask)
        return self.estimate(src_sentemb, mt_sentemb, ref_sentemb)

    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        df = df[["src", "mt", "ref", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["ref"] = df["ref"].astype(str)
        df["score"] = df["score"].astype("float16")
        return df.to_dict("records")

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
    ) -> torch.Tensor:
        """
        Runs one training step and logs the training loss.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.

        :returns: Loss value
        """
        K = self.topk
        batch_input, batch_target = batch
        new_batch_target = []
        batch_prediction = []
        for one_sample, target in zip(batch_input, batch_target['score']):
            prediction = self.forward(**one_sample)
            topk = torch.topk(prediction['score'].view(-1), K).values
            topk_target = target.repeat(K)
            batch_prediction.append(topk)
            new_batch_target.append(topk_target)

        batch_prediction = {'score': torch.hstack(batch_prediction)}
        new_batch_target = {'score': torch.hstack(new_batch_target)}
        loss_value = self.compute_loss(batch_prediction, new_batch_target)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
        return loss_value

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        dataloader_idx: int,
    ) -> None:
        """
        Runs one validation step and logs metrics.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.
        :param dataloader_idx: Integer displaying which dataloader this is.
        """
        K = self.topk
        batch_input, batch_target = batch

        new_batch_target = []
        batch_prediction = []
        for one_sample, target in zip(batch_input, batch_target['score']):
            prediction = self.forward(**one_sample)
            topk = torch.topk(prediction['score'].view(-1), K).values
            topk_target = target.repeat(K)
            batch_prediction.append(topk)
            new_batch_target.append(topk_target)

        batch_prediction = {'score': torch.hstack(batch_prediction)}
        new_batch_target = {'score': torch.hstack(new_batch_target)}
        loss_value = self.compute_loss(batch_prediction, new_batch_target)
        batch_target = new_batch_target
        self.log("val_loss", loss_value, on_step=True, on_epoch=True)

        # TODO: REMOVE if condition after torchmetrics bug fix
        if batch_prediction["score"].view(-1).size() != torch.Size([1]):
            if dataloader_idx == 0:
                self.train_metrics.update(
                    batch_prediction["score"].view(-1), batch_target["score"]
                )
            elif dataloader_idx == 1:
                self.val_metrics.update(
                    batch_prediction["score"].view(-1), batch_target["score"]
                )
