from ..seeker import Seeker
import numpy as np
import torch
import torch.nn as nn
from .fusionComponent import FusionComponent
from .classifier import Classifier
from .semanticFeatureExtractor import SemanticFeatureExtractor
from .statisticalFeatureExtractor import StatisticalFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from glob import glob
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

class SemStaSeeker(Seeker):
    
    def __init__(self, alpha=0.25, semantic_dim=128, statistical_dim=128, num_classes=2, output_size=128, disable_tqdm=False):
        self.fusion_component = FusionComponent(alpha, semantic_dim, statistical_dim)
        self.semantic_feature_extractor = SemanticFeatureExtractor(disable_tqdm=disable_tqdm, output_size=output_size)
        self.statistical_feature_extractor = StatisticalFeatureExtractor(hidden_dim=statistical_dim, disable_tqdm=disable_tqdm)
        self.classifier = Classifier(semantic_dim, num_classes)
        self.disable_tqdm = disable_tqdm

    def train_test_split(self, newsfeeds, labels, test_size=0.2):
        return train_test_split(newsfeeds, labels, test_size=test_size, train_size=1 - test_size, random_state=42)
   
    def evaluate_classifier(self, test_newsfeeds, test_labels):
        self.classifier.eval()
            
        test_newsfeeds = [torch.tensor(inner_list, dtype=torch.float32) for inner_list in test_newsfeeds]
        test_newsfeeds = torch.stack(test_newsfeeds)
        
        test_labels = torch.tensor(test_labels, dtype=torch.float32)

        test_dataset = TensorDataset(test_newsfeeds, test_labels)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                semantic_features = self.semantic_feature_extractor(batch_features)
                statistical_features = self.statistical_feature_extractor.get_statistical_features(batch_features)
                fused_features = self.fusion_component(semantic_features, statistical_features)

                classifier_output = self.classifier(fused_features)
                predicted_labels = torch.round(torch.sigmoid(classifier_output)).squeeze()
                
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='binary')
        recall = recall_score(all_targets, all_predictions, average='binary')
        f1 = f1_score(all_targets, all_predictions, average='binary') 

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    def load_data_from_dir(self, newsfeeds_dir):
        all_newsfeeds = []
        all_labels = []
        benign_feeds = glob(f"{newsfeeds_dir}/*;1")
        stego_feeds = glob(f"{newsfeeds_dir}/*;-1")
        all_feed_paths = sorted(benign_feeds + stego_feeds)
        
        for file_path in all_feed_paths:
            filename = file_path.split('/')[-1]
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)

                    if 'feed' in data and isinstance(data['feed'], list):
                        all_newsfeeds.append(data['feed'])
                        label = filename.split(';')[1]
                        all_labels.append(label)
                    else:
                        print(f"Missing or invalid 'feed' in {filename}")
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except json.JSONDecodeError:
                print(f"Invalid JSON in file: {filename}")
        
        return all_newsfeeds, all_labels

    def train_and_evaluate_model(self, newsfeeds_dir, save_dir='resources/models'):
        all_newsfeeds, all_labels = self.load_data_from_dir(newsfeeds_dir)
        train_newsfeeds, test_newsfeeds, train_labels, test_labels = self.train_test_split(all_newsfeeds, all_labels)

        print('Training AutoEncoder...')
        self.statistical_feature_extractor.train_autoencoder(train_newsfeeds)
        print('AutoEncoder trained')


        print('Evaluate Fused Features of train feeds')
        train_fused_features_tensor = self.compute_fused_features(train_newsfeeds)
        print('Fused Features of train feeds evaluated')

        train_fused_features_save_path = os.path.join(save_dir, 'train_fused_features.pth')
        torch.save(train_fused_features_tensor, train_fused_features_save_path)
        train_data_save_path = os.path.join(save_dir, 'train_data.json')
        with open(train_data_save_path, 'w') as f:
            json.dump({'newsfeeds': train_newsfeeds, 'labels': train_labels}, f)

        print('Evaluate Fused Features of test feeds')
        test_fused_features_tensor = self.compute_fused_features(test_newsfeeds)
        print('Fused Features of test feeds evaluated')

        test_fused_features_save_path = os.path.join(save_dir, 'test_fused_features.pth')
        torch.save(test_fused_features_tensor, test_fused_features_save_path)
        test_data_save_path = os.path.join(save_dir, 'test_data.json')
        with open(test_data_save_path, 'w') as f:
            json.dump({'newsfeeds': test_newsfeeds, 'labels': test_labels}, f)


        print('Training Classifier...')
        self.classifier.train_classifier(train_fused_features_tensor, train_labels)
        print('Classifier trained')
        self.evaluate_classifier(test_fused_features_tensor, test_labels)

    def detect_secret(self, newsfeed: list[str]) -> bool:
        
        self.classifier.eval()
        with torch.no_grad():
            semantic_features = self.semantic_feature_extractor(newsfeed)
            statistical_features = self.statistical_feature_extractor.get_statistical_features(newsfeed)
            fused_features = self.fusion_component(semantic_features, statistical_features)
            classifier_output = self.classifier(fused_features)
            predicted_labels = torch.round(torch.sigmoid(classifier_output)).squeeze()
            print(f"Classification Output after exp conversion: {predicted_labels}")
            decision = classifier_output.item() >= 0.5

        return decision
    
    def compute_fused_features(self, newsfeeds):
        newsfeeds_fused_features_list = []
        
        with torch.no_grad():
            for feed in tqdm(newsfeeds, desc='Evaluate fused features', disable=self.disable_tqdm):
                semantic_features = self.semantic_feature_extractor(feed)
                statistical_features = self.statistical_feature_extractor.get_statistical_features(feed)
                fused_features = self.fusion_component(semantic_features, statistical_features)
                newsfeeds_fused_features_list.append(fused_features)
        
        desired_data_type = torch.float32
        newsfeeds_fused_features_list = [tensor.to(desired_data_type) for tensor in newsfeeds_fused_features_list]

        newsfeeds_fused_features_tensor = torch.stack(newsfeeds_fused_features_list)
        
        return newsfeeds_fused_features_tensor
