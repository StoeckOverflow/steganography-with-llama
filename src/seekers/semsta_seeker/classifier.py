import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, fused_features):
        # Apply attention
        attention_weights = F.softmax(self.attention(fused_features), dim=1)
        attention_applied = fused_features * attention_weights
        
        # Classifier
        logits = self.classifier(attention_applied)
        probabilities = F.log_softmax(logits, dim=1)
        
        return probabilities
