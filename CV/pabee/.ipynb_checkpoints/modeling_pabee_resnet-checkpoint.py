import logging
import os

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from torch.nn import functional as F
from src.transformers.models.resnet.modeling_resnet import (
    ResNetEncoder, 
    ResNetModel, 
    ResNetPreTrainedModel,
)

logger = logging.getLogger(__name__)


def normal_shannon_entropy(p, labels_num):
    entropy = torch.distributions.Categorical(probs = p).entropy()
    normal = -np.log(1.0 / labels_num)
    return entropy / normal

class ResNetEncoderWithPabee(ResNetEncoder):
    def adaptive_forward(self, hidden_state, current_layer):
        """
        Implements the adaptive forward pass for the given layer in ResNet.

        Args:
            hidden_state (torch.Tensor): Input features to the current layer.
            current_layer (int): Index of the current layer.

        Returns:
            torch.Tensor: Output features from the current layer.
        """
        stage_module = self.stages[current_layer]
        hidden_state = stage_module(hidden_state)
        return hidden_state


class ResNetModelWithPabeeVe(ResNetModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ResNetEncoderWithPabee(config)
        self.post_init()
        
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold
        
    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self, to_dir=None):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / len(self.encoder.stages):.2f} ***"
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
        output_dropout=None,
        output_layers=None,
        exiting_threshold=None,
        do_ensemble=False,
        training=False,
    ):
        """
        Forward pass for ResNet with PABEE.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            output_dropout (Callable, optional): Dropout function applied to output layers.
            output_layers (list of nn.Module, optional): List of classification heads for each layer.
            exiting_threshold (float, optional): Entropy threshold for early exit.
            do_ensemble (bool, optional): Whether to perform ensembling on close logits.
            training (bool, optional): Indicates if the model is in training mode.

        Returns:
            List[torch.Tensor]: Outputs from the layers based on PABEE.
        """
        embedding_output = self.embedder(pixel_values)
        hidden_state = embedding_output

        if self.training or self.patience == -1:
            res = []
            for i in range(len(self.encoder.stages)):
                hidden_state = self.encoder.adaptive_forward(hidden_state, current_layer=i)
                pooled_output = self.pooler(hidden_state)
                logits = output_layers[i](output_dropout(pooled_output))
                res.append(logits)
            return res

        elif self.patience == 0:  # Inference with all layers
            for stage_module in self.encoder.stages:
                hidden_state = stage_module(hidden_state)
            pooled_output = self.pooler(hidden_state)
            return [output_layers[-1](pooled_output)]

        else:  # Early exit
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
            
            prev_label = None
            patient_counter = 0

            for i in range(len(self.encoder.stages)):
                calculated_layer_num += 1
                hidden_state = self.encoder.adaptive_forward(hidden_state, current_layer=i)
                pooled_output = self.pooler(hidden_state)
                cur_logits = output_layers[i](pooled_output)
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

class ResNetModelWithPabeeV0(ResNetModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ResNetEncoderWithPabee(config)
        self.post_init()
        
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold
        
    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self, to_dir=None):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / len(self.encoder.stages):.2f} ***"
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
        output_dropout=None,
        output_layers=None,
        exiting_threshold=None,
        do_ensemble=False,
        training=False,
    ):
        """
        Forward pass for ResNet with PABEE.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            output_dropout (Callable, optional): Dropout function applied to output layers.
            output_layers (list of nn.Module, optional): List of classification heads for each layer.
            exiting_threshold (float, optional): Entropy threshold for early exit.
            do_ensemble (bool, optional): Whether to perform ensembling on close logits.
            training (bool, optional): Indicates if the model is in training mode.

        Returns:
            List[torch.Tensor]: Outputs from the layers based on PABEE.
        """
        embedding_output = self.embedder(pixel_values)
        hidden_state = embedding_output

        if self.training or self.patience == -1:
            res = []
            for i in range(len(self.encoder.stages)):
                hidden_state = self.encoder.adaptive_forward(hidden_state, current_layer=i)
                pooled_output = self.pooler(hidden_state)
                logits = output_layers[i](output_dropout(pooled_output))
                res.append(logits)
            return res

        elif self.patience == 0:  # Inference with all layers
            for stage_module in self.encoder.stages:
                hidden_state = stage_module(hidden_state)
            pooled_output = self.pooler(hidden_state)
            return [output_layers[-1](pooled_output)]

        else:  # Early exit
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
            
            prev_label = None
            patient_counter = 0

            for i in range(len(self.encoder.stages)):
                calculated_layer_num += 1
                hidden_state = self.encoder.adaptive_forward(hidden_state, current_layer=i)
                pooled_output = self.pooler(hidden_state)
                cur_logits = output_layers[i](pooled_output)
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

                if patient_counter == self.patience:  #cur_entropy < exiting_threshold: #
                    break

            res = [prev_logits, ]

            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res

class ResNetModelWithPabeeV0e(ResNetModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ResNetEncoderWithPabee(config)
        self.post_init()
        
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold
        
    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self, to_dir=None):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / len(self.encoder.stages):.2f} ***"
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
        output_dropout=None,
        output_layers=None,
        exiting_threshold=None,
        do_ensemble=False,
        training=False,
    ):
        """
        Forward pass for ResNet with PABEE.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            output_dropout (Callable, optional): Dropout function applied to output layers.
            output_layers (list of nn.Module, optional): List of classification heads for each layer.
            exiting_threshold (float, optional): Entropy threshold for early exit.
            do_ensemble (bool, optional): Whether to perform ensembling on close logits.
            training (bool, optional): Indicates if the model is in training mode.

        Returns:
            List[torch.Tensor]: Outputs from the layers based on PABEE.
        """
        embedding_output = self.embedder(pixel_values)
        hidden_state = embedding_output

        if self.training or self.patience == -1:
            res = []
            for i in range(len(self.encoder.stages)):
                hidden_state = self.encoder.adaptive_forward(hidden_state, current_layer=i)
                pooled_output = self.pooler(hidden_state)
                logits = output_layers[i](output_dropout(pooled_output))
                res.append(logits)
            return res

        elif self.patience == 0:  # Inference with all layers
            for stage_module in self.encoder.stages:
                hidden_state = stage_module(hidden_state)
            pooled_output = self.pooler(hidden_state)
            return [output_layers[-1](pooled_output)]

        else:  # Early exit
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
            
            prev_label = None
            patient_counter = 0

            for i in range(len(self.encoder.stages)):
                calculated_layer_num += 1
                hidden_state = self.encoder.adaptive_forward(hidden_state, current_layer=i)
                pooled_output = self.pooler(hidden_state)
                cur_logits = output_layers[i](pooled_output)
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

                if patient_counter == self.patience or cur_entropy < exiting_threshold:
                    break

            res = [prev_logits, ]

            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res

class ResNetForImageClassificationWithPabee(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights_schema = config.weights_schema

        self.config = config

        if config.ee_mechanism == "Ve":
            self.resnet = ResNetModelWithPabeeVe(config)
        elif config.ee_mechanism == "V0":
            self.resnet = ResNetModelWithPabeeV0(config)
        elif config.ee_mechanism == "V0e":
            self.resnet = ResNetModelWithPabeeV0e(config)
        else:
            raise ValueError(f"Unsupported early exiting method: {config.ee_mechanism}!")

        self.dropout = nn.Dropout(0.1)
        self.classifiers = nn.ModuleList(
            nn.Sequential(
                nn.Flatten(),  # Flatten the feature maps to a 2D tensor
                nn.Linear(config.hidden_sizes[i], self.config.num_labels)
            )
            for i in range(len(config.hidden_sizes))
        )

        self.post_init()

    def forward(
        self,
        pixel_values=None,
        labels=None,
    ):
        """
        Forward pass for ResNet with PABEE applied to image classification.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            labels (torch.Tensor, optional): Labels for loss computation.

        Returns:
            Tuple[torch.Tensor]: Loss (if labels provided), logits, and all logits from layers.
        """
        logits = self.resnet(
            pixel_values=pixel_values,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            exiting_threshold=self.config.exiting_threshold,
            do_ensemble=self.config.do_ensemble,
        )

        outputs = (logits[-1], logits)

        if labels is not None:
            total_loss = None
            total_weights = 0
            for ix, logits_item in enumerate(logits):
                if self.num_labels == 1:
#                     logging.info("aaa1")
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits_item.view(-1), labels.view(-1))
                else:
#                     logging.info("bbb1")
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
                        total_loss += loss * (len(self.encoder.stages) - ix - 1)
                    else:
                        total_loss += loss * 1

                if self.weights_schema == "asc":
                    total_weights += ix + 1
                elif self.weights_schema == "desc":
                    total_weights += len(self.encoder.stages) - ix - 1
                else:
                    total_weights += 1

            outputs = (total_loss / total_weights,) + outputs

        return outputs
