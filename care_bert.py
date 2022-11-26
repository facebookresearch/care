# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from transformers import BertModel, BertTokenizer
import torch
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig

class_labels = [
    "adoring",
    "amused",
    "angered",
    "approving",
    "excited",
    "saddened",
    "scared",
]


class CAREBERT(BertPreTrainedModel):
    def __init__(self, config: BertConfig, model_load_path: str = "./care_bert.pth"):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)

        if model_load_path is not None:
            checkpoint = torch.load(model_load_path)
            self.bert.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded from old {model_load_path}")

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Run predictions for a list of texts, returning a list of the list of affects predicted for each example.
def predict(
    examples: List[str], threshold: float = 0.5, model_load_path="./care_bert.pth"
) -> List[List[str]]:
    model = CAREBERT.from_pretrained(
        "bert-base-uncased",
        num_labels=7,
        model_load_path=model_load_path,
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoding = tokenizer(
        examples,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    # forward pass
    outs = model(**encoding, return_dict=False)
    logits = outs[0]
    pred_bools = [pl > threshold for pl in logits]

    predictions = []
    for pred_bool in pred_bools:
        affects = [class_labels[i] for i in range(len(pred_bool)) if pred_bool[i]]
        predictions.append(affects)
    return predictions


if __name__ == "__main__":
    examples = ["Warriors against the Miami Heat!!!", "That was so hilarious"]
    print(predict(examples))
