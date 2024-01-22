from ..seeker import Seeker
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusionComponent import FusionComponent
from .classifier import Classifier
from .semanticFeatureExtractor import SemanticFeatureExtractor
from .statisticalFeatureExtractor import StatisticalFeatureExtractor

class SemStaSeeker(Seeker):
    
    def __init__(self, alpha=0.25, semantic_dim=128, statistical_dim=128, num_classes=2, output_size=128, disable_tqdm=False):
        self.fusion_component = FusionComponent(alpha, semantic_dim, statistical_dim)
        self.semantic_feature_extractor = SemanticFeatureExtractor(disable_tqdm=disable_tqdm, output_size=output_size)
        self.statistical_feature_extractor = StatisticalFeatureExtractor(hidden_dim=statistical_dim, disable_tqdm=disable_tqdm)
        self.classifier = Classifier(semantic_dim, num_classes)
    
    def detect_secret(self, newsfeed: list[str]) -> bool:
        semantic_features = self.semantic_feature_extractor(newsfeed)
        statistical_features = self.statistical_feature_extractor.get_statistical_features(newsfeed)

        fused_features = self.fusion_component(semantic_features, statistical_features)
        classification_output = self.classifier(fused_features)
        probabilities = torch.exp(classification_output)
        print(f"Classification Output after exp conversion: {probabilities}")
        
        not_clean_prob = probabilities[:, 1]
        decision = not_clean_prob > 0.5
        
        return decision.item()