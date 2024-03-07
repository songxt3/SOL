import torch.nn as nn
from transformers import BertModel


class SentimentClassifier(nn.Module):

    def __init__(self, freeze_bert=True, bert_hidden_size=768, num_classes=1):
        super(SentimentClassifier, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('./bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.linear = nn.Linear(in_features=bert_hidden_size, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        # Pass the input through the BERT model
        bert_output = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the output of [CLS] token (representing the whole sequence)
        cls_output = bert_output.last_hidden_state[:, 0, :]

        # Pass the [CLS] token representation through the linear layer
        logits = self.linear(cls_output)

        return logits


class Classifier(nn.Module):
    def __init__(self, bert_hidden_size=768, num_classes=1):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(in_features=bert_hidden_size, out_features=num_classes)

    def forward(self, cls_output):
        logits = self.linear(cls_output)
        return logits


class BERT(nn.Module):

    def __init__(self, freeze_bert=True):
        super(BERT, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('./bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Pass the input through the BERT model
        bert_output = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the output of [CLS] token (representing the whole sequence)
        cls_output = bert_output.last_hidden_state[:, 0, :]

        return cls_output

