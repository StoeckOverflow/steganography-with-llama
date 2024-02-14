import torch
import torch.nn as nn
  
class FusionComponent(nn.Module):
    def __init__(self, semantic_dim, statistical_dim, alpha=0.5):
        super(FusionComponent, self).__init__()
        self.dense = nn.Linear(statistical_dim, semantic_dim)
        #self.alpha = nn.Parameter(torch.full((1,), alpha))
        self.alpha = alpha
        
    def forward(self, semantic_features, statistical_features):
        if self.alpha == 0:
            return semantic_features
        projected_statistical = self.dense(statistical_features)
        low_confidence_mask = (semantic_features >= (0.5 - self.alpha)) & (semantic_features <= (0.5 + self.alpha))
        fused_features = torch.where(low_confidence_mask, projected_statistical, semantic_features)
        return fused_features