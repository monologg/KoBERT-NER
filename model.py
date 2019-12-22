import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, DistilBertModel, PreTrainedModel, DistilBertConfig
from transformers.modeling_distilbert import DistilBertPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class BertClassifier(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(BertClassifier, self).__init__(bert_config)
        self.args = args
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels
        self.slot_classifier = FCLayer(bert_config.hidden_size, bert_config.num_labels, args.dropout_rate, use_activation=False)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]

        logits = self.slot_classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = slot_loss_fct(active_logits, active_labels)
            else:
                loss = slot_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class DistilBertClassifier(DistilBertPreTrainedModel):
    def __init__(self, distilbert_config, args):
        super(DistilBertClassifier, self).__init__(distilbert_config)
        self.args = args
        self.distilbert = DistilBertModel.from_pretrained(args.model_name_or_path, config=distilbert_config)  # Load pretrained distilbert

        self.num_labels = distilbert_config.num_labels
        self.slot_classifier = FCLayer(distilbert_config.hidden_size, distilbert_config.num_labels, args.dropout_rate, use_activation=False)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)  # last-layer hidden-state, (all hidden_states), (all attentions)
        sequence_output = outputs[0]

        logits = self.slot_classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = slot_loss_fct(active_logits, active_labels)
            else:
                loss = slot_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
