# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, The HuggingFace Inc. team and Microsoft Corporation.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch ViT model with Patience-based Early Exit. """


import logging
import os

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from src.transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from src.transformers.models.vit.modeling_vit import ViTModel, ViTConfig, ViTForImageClassification, ViTPreTrainedModel


logger = logging.getLogger(__name__)


def normal_shannon_entropy(p, labels_num):
    entropy = torch.distributions.Categorical(probs = p).entropy()
    normal = -np.log(1.0 / labels_num)
    return entropy / normal


    
class ViTModelWithPabeeVe(ViTModel):
    def __init__(self, config):
        super().__init__(config)

        self.vit = ViTModel(config)

        self.init_weights()
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0
        self.regression_threshold = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold

    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self, to_dir=None):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / self.config.num_hidden_layers:.2f} ***"
        print(message)

        if to_dir is not None:
            if os.path.exists(to_dir):
                with open(to_dir, "r+", encoding="u8") as fp:
                    tmp_data = fp.read()  # 读取所有文件, 文件太大时不用使用此方法
                    fp.seek(0)  # 移动游标
                    fp.write(tmp_data + "\n" + message)
            else:
                with open(to_dir, "w", encoding="u8") as fp:
                    fp.write(message)

        return message

    def forward(
        self,
        pixel_values,
        output_dropout = None,
        output_attentions = None,
        output_layers=None,
        regression=False,
        exiting_threshold=None,
        do_ensemble=False,
        return_dict = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit.embeddings(pixel_values)

        if self.training or self.patience == -1:
            res = []
            for i in range(self.config.num_hidden_layers):
                outputs = self.vit.encoder.layer[i](outputs)[0]
                logits = output_layers[i](output_dropout(outputs[:, 0, :]))
                res.append(logits)

        elif self.patience == 0:  # Use all layers for inference
            for i in range(self.config.num_hidden_layers):
                outputs = self.vit.encoder.layer[i](outputs)[0]
            res = [output_layers[self.config.num_hidden_layers - 1](output_dropout(outputs[:, 0, :]))]
        else:

            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0

            prev_logits = None
            prev_label = None
            prev_entropy = None
            all_prev_logits = []
            all_prev_labels = []

            cur_logits = None
            cur_label = None
            cur_entropy = None

            # 相似度较高的层，进行 ensemble
            all_prev_logits_close = []


            for i in range(self.config.num_hidden_layers):
                calculated_layer_num += 1

                outputs = self.vit.encoder.layer[i](outputs)[0]
                cur_logits = output_layers[i](output_dropout(outputs[:, 0, :]))
                cur_label = cur_logits.detach().argmax(dim=1)

                cur_probs = nn.Softmax(dim=-1)(cur_logits)
                cur_entropy = normal_shannon_entropy(cur_probs, cur_probs.shape[-1])

                if prev_label is not None and torch.all(cur_label.eq(prev_label)):
                    patient_counter += 1
                else:
                    patient_counter = 0

                prev_logits = cur_logits
                prev_label = cur_label

                all_prev_logits.append(cur_logits)

                if cur_entropy < exiting_threshold: #patient_counter == self.patience:
                    break

            res = [prev_logits, ]

            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res



class ViTModelWithPabeeV0(ViTModel):
    def __init__(self, config):
        super().__init__(config)

        self.vit = ViTModel(config)
        self.init_weights()

        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0
        self.regression_threshold = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold

    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self, to_dir=None):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / self.config.num_hidden_layers:.2f} ***"
        print(message)

        if to_dir is not None:
            if os.path.exists(to_dir):
                with open(to_dir, "r+", encoding="u8") as fp:
                    tmp_data = fp.read()  # 读取所有文件, 文件太大时不用使用此方法
                    fp.seek(0)  # 移动游标
                    fp.write(tmp_data + "\n" + message)
            else:
                with open(to_dir, "w", encoding="u8") as fp:
                    fp.write(message)

        return message

    def forward(
        self,
        pixel_values,
        output_dropout = None,
        output_attentions = None,
        output_layers=None,
        regression=False,
        exiting_threshold=None,
        do_ensemble=False,
        return_dict = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit.embeddings(pixel_values)

        if self.training or self.patience == -1:
            res = []
            for i in range(self.config.num_hidden_layers):
                outputs = self.vit.encoder.layer[i](outputs)[0]
                logits = output_layers[i](output_dropout(outputs[:, 0, :]))
                res.append(logits)

        elif self.patience == 0:  # Use all layers for inference
            for i in range(self.config.num_hidden_layers):
                outputs = self.vit.encoder.layer[i](outputs)[0]
            res = [output_layers[self.config.num_hidden_layers - 1](output_dropout(outputs[:, 0, :]))]
        else:

            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0

            prev_logits = None
            prev_label = None
            prev_entropy = None
            all_prev_logits = []
            all_prev_labels = []

            cur_logits = None
            cur_label = None
            cur_entropy = None

            # 相似度较高的层，进行 ensemble
            all_prev_logits_close = []


            for i in range(self.config.num_hidden_layers):
                calculated_layer_num += 1

                outputs = self.vit.encoder.layer[i](outputs)[0]
                cur_logits = output_layers[i](output_dropout(outputs[:, 0, :]))
                cur_label = cur_logits.detach().argmax(dim=1)

                cur_probs = nn.Softmax(dim=-1)(cur_logits)
                cur_entropy = normal_shannon_entropy(cur_probs, cur_probs.shape[-1])

                if prev_label is not None and torch.all(cur_label.eq(prev_label)):
                    patient_counter += 1
                else:
                    patient_counter = 0

                prev_logits = cur_logits
                prev_label = cur_label

                all_prev_logits.append(cur_logits)

                if patient_counter == self.patience:
                    break

            res = [prev_logits, ]

            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res

class ViTModelWithPabeeV0e(ViTModel):
    def __init__(self, config):
        super().__init__(config)

        self.vit = ViTModel(config)

        self.init_weights()
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0
        self.regression_threshold = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold

    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self, to_dir=None):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / self.config.num_hidden_layers:.2f} ***"
        print(message)

        if to_dir is not None:
            if os.path.exists(to_dir):
                with open(to_dir, "r+", encoding="u8") as fp:
                    tmp_data = fp.read()  # 读取所有文件, 文件太大时不用使用此方法
                    fp.seek(0)  # 移动游标
                    fp.write(tmp_data + "\n" + message)
            else:
                with open(to_dir, "w", encoding="u8") as fp:
                    fp.write(message)

        return message

    def forward(
        self,
        pixel_values,
        output_dropout = None,
        output_layers=None,
        regression=False,
        exiting_threshold=None,
        do_ensemble=False,
        return_dict = None,
    ):
        # print("Pixel values shape:", pixel_values.shape)
        # exit()

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit.embeddings(pixel_values)
        # print(outputs.shape)
        # print(outputs[0].size())
        # exit()

        if self.training or self.patience == -1:
            res = []
            for i in range(self.config.num_hidden_layers):
                outputs = self.vit.encoder.layer[i](outputs)[0]
                logits = output_layers[i](output_dropout(outputs[:, 0, :]))
                res.append(logits)

        elif self.patience == 0:  # Use all layers for inference
            for i in range(self.config.num_hidden_layers):
                outputs = self.vit.encoder.layer[i](outputs)[0]
            res = [output_layers[self.config.num_hidden_layers - 1](output_dropout(outputs[:, 0, :]))]
        else:

            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0

            prev_logits = None
            prev_label = None
            prev_entropy = None
            all_prev_logits = []
            all_prev_labels = []

            cur_logits = None
            cur_label = None
            cur_entropy = None

            # 相似度较高的层，进行 ensemble
            all_prev_logits_close = []


            for i in range(self.config.num_hidden_layers):
                calculated_layer_num += 1

                outputs = self.vit.encoder.layer[i](outputs)[0]
                cur_logits = output_layers[i](output_dropout(outputs[:, 0, :]))
                cur_label = cur_logits.detach().argmax(dim=1)

                cur_probs = nn.Softmax(dim=-1)(cur_logits)
                cur_entropy = normal_shannon_entropy(cur_probs, cur_probs.shape[-1])

                if prev_label is not None and torch.all(cur_label.eq(prev_label)):
                    patient_counter += 1
                else:
                    patient_counter = 0

                prev_logits = cur_logits
                prev_label = cur_label

                all_prev_logits.append(cur_logits)

                if cur_entropy < exiting_threshold or patient_counter == self.patience:
                    break

            res = [prev_logits, ]

            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res
#####################################################################################################################################################################


class ViTForSequenceClassificationWithPabee(ViTPreTrainedModel):
    def __init__(self, config,):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights_schema = config.weights_schema

        self.config = config

        if config.ee_mechanism == "V0":
            self.vit = ViTModelWithPabeeV0(config)
        elif config.ee_mechanism == "V0e":
            self.vit = ViTModelWithPabeeV0e(config)
        elif config.ee_mechanism == "Ve":
            self.vit = ViTModelWithPabeeVe(config)
        elif config.ee_mechanism == "V1":
            self.vit = ViTModelWithPabeeV1(config)
        elif config.ee_mechanism == "V2":
            self.vit = ViTModelWithPabeeV2(config)
        else:
            raise ValueError(
                f"unspported early exiting method: {config.ee_mechanism} !!! "
            )


        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.hidden_size, self.config.num_labels) for _ in range(config.num_hidden_layers)]
        )

        self.init_weights() 

    def forward(
        self,
        pixel_values=None,
        labels=None,
    ):

        logits = self.vit(
            pixel_values=pixel_values,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            regression=self.num_labels == 1,
            exiting_threshold=self.config.exiting_threshold,
            do_ensemble=self.config.do_ensemble,
        )

        outputs = (logits[-1], logits)

        if labels is not None:
            total_loss = None
            total_weights = 0
            for ix, logits_item in enumerate(logits):
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits_item.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
#                     logging.info(f"logits_item shape: {logits_item.shape}")
#                     logging.info(f"logits_item: {logits_item}")
#                     logging.info(f"labels shape: {labels.shape}")
#                     logging.info(f"labels: {labels}")
                    loss = loss_fct(logits_item.view(-1, self.num_labels), labels.view(-1))
                if total_loss is None:
                    total_loss = loss
                else:
                    if self.weights_schema == "asc":
                        total_loss += loss * (ix + 1)
                    elif self.weights_schema == "desc":
                        total_loss += loss * (self.config.num_hidden_layers - ix - 1)
                    else:
                        total_loss += loss * 1

                if self.weights_schema == "asc":
                    total_weights += ix + 1
                elif self.weights_schema == "desc":
                    total_weights += self.config.num_hidden_layers - ix - 1
                else:
                    total_weights += 1

            outputs = (total_loss / total_weights,) + outputs

        return outputs
