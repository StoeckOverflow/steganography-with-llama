from ..seeker import Seeker
import torch
from .fusionComponent import FusionComponent
from .classifier import Classifier, Classifier_Trainer
from .semanticFeatureExtractor import SemanticFeatureExtractor
from .statisticalFeatureExtractor import StatisticalFeatureExtractor
from sklearn.model_selection import train_test_split
from glob import glob
import json
from tqdm import tqdm
import os

class SemStaSeeker(Seeker):
    
    def __init__(self, alpha=0.25, semantic_dim=1024, statistical_dim=1024, num_classes=2, output_size=1024, disable_tqdm=False):
        self.fusion_component = FusionComponent(semantic_dim, statistical_dim, alpha)
        self.semantic_feature_extractor = SemanticFeatureExtractor(output_size=output_size)
        self.statistical_feature_extractor = StatisticalFeatureExtractor(hidden_dim=statistical_dim, output_dim=output_size, disable_tqdm=disable_tqdm)
        self.classifier = Classifier(statistical_dim, semantic_dim, num_classes)
        self.classifier_trainer = Classifier_Trainer(self.classifier)
        self.disable_tqdm = disable_tqdm

    def train_test_split(self, newsfeeds, labels, test_size=0.3):
        return train_test_split(newsfeeds, labels, test_size=test_size, train_size=1 - test_size, random_state=42, stratify=labels)
        
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

    def train_and_evaluate_model_saved_data(self):
        with open('resources/models/train_data.json', 'r') as json_file:
            train_data = json.load(json_file)
        train_newsfeeds = train_data['newsfeeds']
        train_labels = train_data['labels']
        
        with open('resources/models/test_data.json', 'r') as json_file:
            test_data = json.load(json_file)
        test_newsfeeds = test_data['newsfeeds']
        test_labels = test_data['labels']
                
        print('Training AutoEncoder...')
        self.statistical_feature_extractor.train_autoencoder(train_newsfeeds)
        print('AutoEncoder trained')

        print('Evaluate Fused Features of train feeds')
        train_fused_features_tensor = torch.load('resources/models/train_fused_features.pth')
        train_original_encoder_features_tensor = torch.load('resources/models/train_original_encoder_features.pth')
        print('Fused Features of train feeds evaluated')

        print('Evaluate Fused Features of test feeds')
        test_fused_features_tensor = torch.load('resources/models/test_fused_features.pth')
        test_original_encoder_features_tensor = torch.load('resources/models/test_original_encoder_features.pth')
        print('Fused Features of test feeds evaluated')

        print('Training Classifier...')
        self.classifier_trainer.train_classifier_with_cross_validation(train_original_encoder_features_tensor, train_fused_features_tensor, train_labels)
        print('Classifier trained')
        print('Evaluate Classifier')
        self.classifier_trainer.evaluate_classifier(test_original_encoder_features_tensor, test_fused_features_tensor, test_labels)
        
        print('Evaluate Classifier on Stemo Newsfeeds')
        all_newsfeeds_stemo, all_labels_stemo = self.load_data_from_dir('resources/feeds/doctored_feeds_newsfeeds')
        stemo_newsfeeds_fused_features_tensor, stemo_newsfeeds_original_encoder_features_tensor = self.compute_fused_and_original_encoder_features(all_newsfeeds_stemo)
        self.classifier_trainer.evaluate_classifier(stemo_newsfeeds_original_encoder_features_tensor, stemo_newsfeeds_fused_features_tensor, all_labels_stemo)
        

    def train_and_evaluate_model(self, newsfeeds_dir, save_dir='resources/models'):
        all_newsfeeds, all_labels = self.load_data_from_dir(newsfeeds_dir)
        train_newsfeeds, test_newsfeeds, train_labels, test_labels = self.train_test_split(all_newsfeeds, all_labels)
        
        print('Training AutoEncoder...')
        self.statistical_feature_extractor.train_autoencoder(train_newsfeeds)
        print('AutoEncoder trained')

        print('Evaluate Fused Features of train feeds')
        train_fused_features_tensor, train_original_encoder_features_tensor = self.compute_fused_and_original_encoder_features(train_newsfeeds)
        print('Fused Features of train feeds evaluated')

        train_fused_features_save_path = os.path.join(save_dir, 'train_fused_features.pth')
        torch.save(train_fused_features_tensor, train_fused_features_save_path)
        train_original_encoder_features_save_path = os.path.join(save_dir, 'train_original_encoder_features.pth')
        torch.save(train_original_encoder_features_tensor, train_original_encoder_features_save_path)
        train_data_save_path = os.path.join(save_dir, 'train_data.json')
        with open(train_data_save_path, 'w') as f:
            json.dump({'newsfeeds': train_newsfeeds, 'labels': train_labels}, f)

        print('Evaluate Fused Features of test feeds')
        test_fused_features_tensor, test_original_encoder_features_tensor = self.compute_fused_and_original_encoder_features(test_newsfeeds)
        print('Fused Features of test feeds evaluated')

        test_fused_features_save_path = os.path.join(save_dir, 'test_fused_features.pth')
        torch.save(test_fused_features_tensor, test_fused_features_save_path)
        test_original_encoder_features_save_path = os.path.join(save_dir, 'test_original_encoder_features.pth')
        torch.save(test_original_encoder_features_tensor, test_original_encoder_features_save_path)
        test_data_save_path = os.path.join(save_dir, 'test_data.json')
        with open(test_data_save_path, 'w') as f:
            json.dump({'newsfeeds': test_newsfeeds, 'labels': test_labels}, f)

        print('Training Classifier...')
        self.classifier_trainer.train_classifier_with_cross_validation(train_original_encoder_features_tensor, train_fused_features_tensor, train_labels)
        print('Classifier trained')
        self.classifier_trainer.evaluate_classifier(test_original_encoder_features_tensor, test_fused_features_tensor, test_labels)

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
    
    def compute_fused_and_original_encoder_features(self, newsfeeds):
        newsfeeds_fused_features_list = []
        newsfeeds_original_encoder_features_list = []
        with torch.no_grad():
            for feed in tqdm(newsfeeds, desc='Evaluate fused features', disable=self.disable_tqdm):
                semantic_features = self.semantic_feature_extractor(feed)
                statistical_features = self.statistical_feature_extractor.get_statistical_features(feed)
                fused_features = self.fusion_component(semantic_features, statistical_features)
                newsfeeds_fused_features_list.append(fused_features)
                newsfeeds_original_encoder_features_list.append(semantic_features)
        
        desired_data_type = torch.float32
        newsfeeds_fused_features_list = [tensor.to(desired_data_type) for tensor in newsfeeds_fused_features_list]
        newsfeeds_fused_features_tensor = torch.stack(newsfeeds_fused_features_list)
        
        newsfeeds_original_encoder_features_list = [tensor.to(desired_data_type) for tensor in newsfeeds_original_encoder_features_list]
        newsfeeds_original_encoder_features_tensor = torch.stack(newsfeeds_original_encoder_features_list)
        
        return newsfeeds_fused_features_tensor, newsfeeds_original_encoder_features_tensor