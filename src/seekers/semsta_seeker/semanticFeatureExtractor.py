from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class SemanticFeatureExtractor(nn.Module):
    def __init__(self, output_size, bert_model_name='bert-base-cased'):
        super(SemanticFeatureExtractor, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.dense = nn.Linear(self.bert_model.config.hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, newsfeed):
        inputs = self.bert_tokenizer(newsfeed, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        dense_output = self.dense(cls_representation)
        confidence_scores = self.sigmoid(dense_output)

        return confidence_scores