from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class SemanticFeatureExtractor(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', output_size=1024):
        super(SemanticFeatureExtractor, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.dense = nn.Linear(self.bert_model.config.hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, newsfeed):
        inputs = self.bert_tokenizer(newsfeed, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_layer_output = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

        # Apply mean pooling across the sequence length dimension
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_layer_output.size()).float()
        sum_embeddings = torch.sum(last_layer_output * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)  # This avoids division by zero
        mean_pooled_output = sum_embeddings / sum_mask

        dense_output = self.dense(mean_pooled_output)  # Now shape: (batch_size, output_size)
        confidence_scores = self.sigmoid(dense_output)
        return confidence_scores