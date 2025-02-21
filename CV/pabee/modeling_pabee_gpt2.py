"""PyTorch GPT2 model with Patience-based Early Exit. """

import logging
import os

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from src.transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel

logger = logging.getLogger(__name__)


def normal_shannon_entropy(p, labels_num):
    entropy = torch.distributions.Categorical(probs=p).entropy()
    normal = -torch.log(torch.tensor(1.0 / labels_num))
    return entropy / normal


class GPT2ModelWithPabeeVe(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gpt2 = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出层

        self.init_weights()

        # Early Exit 参数
        self.patience = 0
        self.exiting_threshold = 0.0
        self.inference_layers_num = 0
        self.inference_instances_num = 0
    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold
        
    def set_patience(self, patience):
        self.patience = patience

    def set_exiting_threshold(self, threshold):
        self.exiting_threshold = threshold

    def reset_stats(self):
        self.inference_layers_num = 0
        self.inference_instances_num = 0

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

    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        block = self.gpt2.h[current_layer]  # 获取指定的层
        outputs = block(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask[current_layer] if head_mask is not None else None,
        )
        return outputs[0]  # 返回当前层的隐藏状态

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_dropout=None,
        output_layers=None,
        regression=False,
        exiting_threshold=None,
        do_ensemble=False,
    ):

        # 初始化输入
        input_shape = input_ids.size()
        device = input_ids.device

        attention_mask = attention_mask if attention_mask is not None else torch.ones(input_shape, device=device)
        extended_attention_mask = self.gpt2.get_extended_attention_mask(attention_mask, input_shape, device)

        hidden_states = self.gpt2.wte(input_ids) + self.gpt2.wpe(torch.arange(input_ids.size(1), device=device))
        hidden_states = self.gpt2.drop(hidden_states)
#         logging.info(f"hidden_states.size: {hidden_states.size()}")

        # Early Exit 初始化
        patient_counter = 0
        prev_logits = None
        all_logits = []
        all_hidden_states = [] if encoder_hidden_states else None
        if self.training or self.patience == -1:
            res = []
            for i in range(self.config.n_layer):
                hidden_states = self.adaptive_forward(hidden_states, current_layer=i, attention_mask=extended_attention_mask, head_mask=None)
                if len(hidden_states.size())>2:
                    logits = output_layers[i](output_dropout(hidden_states[:,-1,:]))
                else:
                    logits = output_layers[i](output_dropout(hidden_states))
#                 logging.info(f"logitssize: {logits.size()}")
#                 logging.info(f"logits: {logits}")
                res.append(logits)
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
            
            for i in range(self.config.n_layer):
                calculated_layer_num += 1
                hidden_states = self.adaptive_forward(
                    hidden_states, current_layer=i, attention_mask=extended_attention_mask, head_mask=None
                )
                if len(hidden_states.size())>2:
                    cur_logits = output_layers[i](output_dropout(hidden_states[:,-1,:]))
                else:
                    cur_logits = output_layers[i](output_dropout(hidden_states))

#                 cur_logits = output_layers[i](hidden_states)
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


class GPT2ModelWithPabeeV0(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gpt2 = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出层

        self.init_weights()

        # Early Exit 参数
        self.patience = 0
        self.exiting_threshold = 0.0
        self.inference_layers_num = 0
        self.inference_instances_num = 0
    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold
        
    def set_patience(self, patience):
        self.patience = patience

    def set_exiting_threshold(self, threshold):
        self.exiting_threshold = threshold

    def reset_stats(self):
        self.inference_layers_num = 0
        self.inference_instances_num = 0

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

    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        block = self.gpt2.h[current_layer]  # 获取指定的层
        outputs = block(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask[current_layer] if head_mask is not None else None,
        )
        return outputs[0]  # 返回当前层的隐藏状态

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_dropout=None,
        output_layers=None,
        regression=False,
        exiting_threshold=None,
        do_ensemble=False,
    ):

        # 初始化输入
        input_shape = input_ids.size()
        device = input_ids.device

        attention_mask = attention_mask if attention_mask is not None else torch.ones(input_shape, device=device)
        extended_attention_mask = self.gpt2.get_extended_attention_mask(attention_mask, input_shape, device)

        hidden_states = self.gpt2.wte(input_ids) + self.gpt2.wpe(torch.arange(input_ids.size(1), device=device))
        hidden_states = self.gpt2.drop(hidden_states)
#         logging.info(f"hidden_states.size: {hidden_states.size()}")

        # Early Exit 初始化
        patient_counter = 0
        prev_logits = None
        all_logits = []
        all_hidden_states = [] if encoder_hidden_states else None
        if self.training or self.patience == -1:
            res = []
            for i in range(self.config.n_layer):
                hidden_states = self.adaptive_forward(hidden_states, current_layer=i, attention_mask=extended_attention_mask, head_mask=None)
                if len(hidden_states.size())>2:
                    logits = output_layers[i](output_dropout(hidden_states[:,-1,:]))
                else:
                    logits = output_layers[i](output_dropout(hidden_states))
#                 logging.info(f"logitssize: {logits.size()}")
#                 logging.info(f"logits: {logits}")
                res.append(logits)
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
            
            for i in range(self.config.n_layer):
                calculated_layer_num += 1
                hidden_states = self.adaptive_forward(
                    hidden_states, current_layer=i, attention_mask=extended_attention_mask, head_mask=None
                )
                if len(hidden_states.size())>2:
                    cur_logits = output_layers[i](output_dropout(hidden_states[:,-1,:]))
                else:
                    cur_logits = output_layers[i](output_dropout(hidden_states))

#                 cur_logits = output_layers[i](hidden_states)
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


class GPT2ModelWithPabeeV0e(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gpt2 = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出层

        self.init_weights()

        # Early Exit 参数
        self.patience = 0
        self.exiting_threshold = 0.0
        self.inference_layers_num = 0
        self.inference_instances_num = 0
    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold
        
    def set_patience(self, patience):
        self.patience = patience

    def set_exiting_threshold(self, threshold):
        self.exiting_threshold = threshold

    def reset_stats(self):
        self.inference_layers_num = 0
        self.inference_instances_num = 0

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

    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        block = self.gpt2.h[current_layer]  # 获取指定的层
        outputs = block(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask[current_layer] if head_mask is not None else None,
        )
        return outputs[0]  # 返回当前层的隐藏状态

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_dropout=None,
        output_layers=None,
        regression=False,
        exiting_threshold=None,
        do_ensemble=False,
    ):

        # 初始化输入
        input_shape = input_ids.size()
        device = input_ids.device

        attention_mask = attention_mask if attention_mask is not None else torch.ones(input_shape, device=device)
        extended_attention_mask = self.gpt2.get_extended_attention_mask(attention_mask, input_shape, device)

        hidden_states = self.gpt2.wte(input_ids) + self.gpt2.wpe(torch.arange(input_ids.size(1), device=device))
        hidden_states = self.gpt2.drop(hidden_states)
#         logging.info(f"hidden_states.size: {hidden_states.size()}")

        # Early Exit 初始化
        patient_counter = 0
        prev_logits = None
        all_logits = []
        all_hidden_states = [] if encoder_hidden_states else None
        if self.training or self.patience == -1:
            res = []
            for i in range(self.config.n_layer):
                hidden_states = self.adaptive_forward(hidden_states, current_layer=i, attention_mask=extended_attention_mask, head_mask=None)
                if len(hidden_states.size())>2:
                    logits = output_layers[i](output_dropout(hidden_states[:,-1,:]))
                else:
                    logits = output_layers[i](output_dropout(hidden_states))
#                 logging.info(f"logitssize: {logits.size()}")
#                 logging.info(f"logits: {logits}")
                res.append(logits)
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
            
            for i in range(self.config.n_layer):
                calculated_layer_num += 1
                hidden_states = self.adaptive_forward(
                    hidden_states, current_layer=i, attention_mask=extended_attention_mask, head_mask=None
                )
                if len(hidden_states.size())>2:
                    cur_logits = output_layers[i](output_dropout(hidden_states[:,-1,:]))
                else:
                    cur_logits = output_layers[i](output_dropout(hidden_states))

#                 cur_logits = output_layers[i](hidden_states)
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
# =============================================================
    
class Gpt2ForSequenceClassificationWithPabee(GPT2PreTrainedModel):
    def __init__(self, config,):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights_schema = config.weights_schema

        self.config = config

        if config.ee_mechanism == "Ve":
            self.gpt2 = GPT2ModelWithPabeeVe(config)
        elif config.ee_mechanism == "V0":
            self.gpt2 = GPT2ModelWithPabeeV0(config)
        elif config.ee_mechanism == "V0e":
            self.gpt2 = GPT2ModelWithPabeeV0e(config)
        else:
            raise ValueError(
                f"unspported early exiting method: {config.ee_mechanism} !!! "
            )

        print(config)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.hidden_size, self.config.num_labels) for _ in range(config.num_hidden_layers)]
        )

        self.init_weights() 

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification (or regression if config.num_labels==1) loss.
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

        Examples::

            from transformers import BertTokenizer, BertForSequenceClassification
            from pabee import BertForSequenceClassificationWithPabee
            from torch import nn
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassificationWithPabee.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)

            loss, logits = outputs[:2]

        """

        logits = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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
#                     logging.info(f"self.num_labels: {self.num_labels}")
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
