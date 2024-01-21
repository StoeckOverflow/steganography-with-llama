from ..seeker import Seeker
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusionComponent import FusionComponent
from .classifier import Classifier

class SemStaSeeker(nn.Module):
    def __init__(self, alpha, semantic_dim, statistical_dim, num_classes):
        super(SemStaSeeker, self).__init__()
        self.fusion_component = FusionComponent(alpha, semantic_dim, statistical_dim)
        self.classifier = Classifier(semantic_dim, num_classes)
    
    def forward(self, semantic_features, statistical_features):
        fused_features = self.fusion_component(semantic_features, statistical_features)
        classification_output = self.classifier(fused_features)
        return classification_output
