from transformers import BertTokenizer, BertModel
import torch
from torch import nn

class SemanticFeatureExtractor(nn.Module):
    
    def __init__(self, output_size, disable_tqdm=False):
        super(SemanticFeatureExtractor, self).__init__()
        self.disable_tqdm=False
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.dense = nn.Linear(self.bert_model.config.hidden_size, output_size)
        self.activation = nn.Sigmoid()
        
    def forward(self, newsfeed):
        inputs = self.tokenizer(newsfeed, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, 1)
        dense_output = self.dense(pooled_output)
        activated_output = self.activation(dense_output)
        
        return activated_output