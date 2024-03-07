import torch.nn as nn
from transformers import BertModel


class MRPCModel(nn.Module):
    def __init__(self, config):
        super(MRPCModel, self).__init__()
        self.config = config

        # 使用预训练的BERT模型
        self.bert = BertModel.from_pretrained('./bert-base-uncased')
        for p in self.bert.parameters():
            p.requires_grad = False

        # 构建其他层
        self.dense = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # BERT的前向传播
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']

        # 添加其他层
        dense_output = self.dense(pooled_output)
        preds = self.sigmoid(dense_output)

        return preds


class BERT(nn.Module):

    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config

        # Using a pre-trained BERT model
        self.bert = BertModel.from_pretrained('./bert-base-uncased')
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Forward propagation of BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']

        return pooled_output


class Classifier(nn.Module):
    def __init__(self, bert_hidden_size=768, num_classes=1):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(in_features=bert_hidden_size, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cls_output):
        dense_output = self.linear(cls_output)
        preds = self.sigmoid(dense_output)
        return preds