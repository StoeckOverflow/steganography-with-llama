import torch
import torch.nn as nn

class FusionComponent(nn.Module):
    def __init__(self, alpha, semantic_dim, statistical_dim):
        super(FusionComponent, self).__init__()
        self.alpha = alpha
        self.dense = nn.Linear(statistical_dim, semantic_dim)

    def forward(self, semantic_features, statistical_features):
        projected_statistical = self.dense(statistical_features)
        
        low_confidence_mask = (semantic_features < 0.5 - self.alpha) | (semantic_features > 0.5 + self.alpha)
        high_confidence_semantic = torch.where(low_confidence_mask, semantic_features, torch.tensor(0.5).to(semantic_features.device))
        
        fused_features = torch.where(low_confidence_mask, high_confidence_semantic, projected_statistical)
        
        return fused_features