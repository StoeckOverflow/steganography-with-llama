from ..seeker import Seeker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusionComponent import FusionComponent
from .classifier import Classifier
from .semanticFeatureExtractor import SemanticFeatureExtractor
from .statisticalFeatureExtractor import StatisticalFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import glob
import json

class SemStaSeeker(Seeker):
    
    def __init__(self, alpha=0.25, semantic_dim=128, statistical_dim=128, num_classes=2, output_size=128, disable_tqdm=False):
        self.fusion_component = FusionComponent(alpha, semantic_dim, statistical_dim)
        self.semantic_feature_extractor = SemanticFeatureExtractor(disable_tqdm=disable_tqdm, output_size=output_size)
        self.statistical_feature_extractor = StatisticalFeatureExtractor(hidden_dim=statistical_dim, disable_tqdm=disable_tqdm)
        self.classifier = Classifier(semantic_dim, num_classes)

    def train_test_split(self, newsfeeds, labels, test_size=0.2):
        return train_test_split(newsfeeds, labels, test_size=test_size, random_state=42)

    def train_classifier(self, train_newsfeeds, train_labels, num_epochs=10, learning_rate=0.001, batch_size=32):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(num_epochs):
            total_loss = 0.0
            # Shuffle and create batches
            permutation = torch.randperm(len(train_newsfeeds))
            for i in range(0, len(train_newsfeeds), batch_size):
                indices = permutation[i:i + batch_size]
                batch_newsfeeds = [train_newsfeeds[j] for j in indices]
                batch_labels = torch.tensor([train_labels[j] for j in indices], dtype=torch.float32)

                batch_semantic_features = self.semantic_feature_extractor(batch_newsfeeds)
                batch_statistical_features = self.statistical_feature_extractor.get_statistical_features(batch_newsfeeds)
                fused_features = self.fusion_component(batch_semantic_features, batch_statistical_features)

                classifier_output = self.classifier(fused_features)
                loss = criterion(classifier_output, torch.tensor([batch_labels], dtype=torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(permutation):.4f}')
    
    def evaluate_classifier(self, test_newsfeeds, test_labels):
        self.classifier.eval()
        predictions = []
        
        with torch.no_grad():
            for newsfeed in test_newsfeeds:
                semantic_features = self.semantic_feature_extractor(newsfeed)
                statistical_features = self.statistical_feature_extractor.get_statistical_features(newsfeed)
                fused_features = self.fusion_component(semantic_features, statistical_features)

                classifier_output = self.classifier(fused_features)
                predicted_label = torch.round(torch.sigmoid(classifier_output)).item()
                predictions.append(predicted_label)
        
        predictions = [int(p) for p in predictions]
        test_labels = [int(label) for label in test_labels]

        accuracy = sum([pred == label for pred, label in zip(predictions, test_labels)]) / len(test_labels)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        self.classifier.train()

    def load_data_from_dir(newsfeeds_dir):
        all_newsfeeds = []
        all_labels = []

        for file_path in glob.glob(f"{newsfeeds_dir}/*.json"):
            filename = file_path.split('/')[-1]
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)

                    if 'feed' in data and isinstance(data['feed'], list):
                        all_newsfeeds.extend(data['feed'])
                        label = filename.split(';')[1]
                        all_labels.append(label)
                    else:
                        print(f"Missing or invalid 'feed' in {filename}")
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except json.JSONDecodeError:
                print(f"Invalid JSON in file: {filename}")
        
        return all_newsfeeds, all_labels

    def train_and_evaluate_model(self, newsfeeds_dir):
        all_newsfeeds, all_labels = self.load_data_from_dir(newsfeeds_dir)
        train_newsfeeds, test_newsfeeds, train_labels, test_labels = self.train_test_split(all_newsfeeds, all_labels)
        self.statistical_feature_extractor.train_autoencoder(train_newsfeeds)
        self.train_classifier(train_newsfeeds, train_labels)
        self.evaluate_classifier(test_newsfeeds, test_labels)
    
    def detect_secret(self, newsfeed: list[str]) -> bool:
        semantic_features = self.semantic_feature_extractor(newsfeed)
        statistical_features = self.statistical_feature_extractor.get_statistical_features(newsfeed)

        fused_features = self.fusion_component(semantic_features, statistical_features)
        classification_output = self.classifier(fused_features)
        probabilities = torch.exp(classification_output)
        print(f"Classification Output after exp conversion: {probabilities}")
        
        not_clean_prob = probabilities[:, 1]
        decision = np.mean(not_clean_prob.detach().numpy()) > 0.5
        #bools = not_clean_prob > 0.5
        #decision = bools.any()
        
        return decision