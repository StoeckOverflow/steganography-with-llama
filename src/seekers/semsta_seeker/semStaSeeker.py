from ..seeker import Seeker
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusionComponent import FusionComponent
from .classifier import Classifier
from .semanticFeatureExtractor import SemanticFeatureExtractor
from .statisticalFeatureExtractor import StatisticalFeatureExtractor

class SemStaSeeker(Seeker):
    def __init__(self, alpha, semantic_dim, statistical_dim, num_classes, output_size):
        self.fusion_component = FusionComponent(alpha, semantic_dim, statistical_dim)
        self.semantic_feature_extractor = SemanticFeatureExtractor(output_size)
        self.statistical_feature_extractor = StatisticalFeatureExtractor()
        self.classifier = Classifier(semantic_dim, num_classes)
    
    def get_classification(self, semantic_features, statistical_features):
        fused_features = self.fusion_component(semantic_features, statistical_features)
        classification_output = self.classifier(fused_features)
        return classification_output

    def detect_secret(self, newsfeed: list[str]) -> bool:
        
        semantic_features = self.semantic_feature_extractor(newsfeed)
        statistical_features = self.statistical_feature_extractor(newsfeed)
        fused_features = self.fusion_component(semantic_features, statistical_features)
        classification_output = self.classifier(fused_features)
        print(f"Classification Output: {classification_output}")
        
        #Implement the Decision Logic here
        
        return True