from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from glob import glob
import json
import torch
import os
from ..seeker import Seeker
from .fusionComponent import FusionComponent
from .classifier import Classifier, Classifier_Trainer
from .semanticFeatureExtractor import SemanticFeatureExtractor
from .statisticalFeatureExtractor import StatisticalFeatureExtractor
from .data_augmentor import DataAugmentor
import random

class SemStaSeeker(Seeker):
    def __init__(self, alpha=0.25, semantic_dim=1024, statistical_dim=1024, num_classes=2, output_size=1024, disable_tqdm=False):
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.fusion_component = FusionComponent(semantic_dim, statistical_dim, alpha)
        self.semantic_feature_extractor = SemanticFeatureExtractor(output_size=output_size)
        self.statistical_feature_extractor = StatisticalFeatureExtractor(hidden_dim=statistical_dim, output_dim=output_size, disable_tqdm=disable_tqdm)
        self.classifier = Classifier(statistical_dim, semantic_dim, num_classes)
        self.classifier_trainer = Classifier_Trainer(self.classifier)
        self.disable_tqdm = disable_tqdm

    def train_test_split(self, newsfeeds, labels, test_size=0.2):
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

    def train_and_evaluate_model_on_saved_data(self, save_dir='resources/models'):
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
        avg_metrics, best_f1_score, best_model = self.classifier_trainer.train_classifier_with_cross_validation(train_original_encoder_features_tensor, train_fused_features_tensor, train_labels)
        self.classifier.load_state_dict(best_model)
        torch.save(self.classifier.state_dict(), os.path.join(save_dir,'best_classifier_crossvalidation.pth'))
        print('Classifier trained')
        print('Evaluate Classifier')
        self.classifier_trainer.evaluate_classifier(test_original_encoder_features_tensor, test_fused_features_tensor, test_labels)
        
    def train_and_evaluate_model(self, newsfeeds_dir, save_dir='resources/models'):
        train_newsfeeds, train_labels = self.load_data_from_dir(newsfeeds_dir)
        test_newsfeeds, test_labels = self.load_data_from_dir('resources/feeds/doctored_feeds_newsfeeds')
        
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
        avg_metrics, best_f1_score, best_model = self.classifier_trainer.train_classifier_with_cross_validation(train_original_encoder_features_tensor, train_fused_features_tensor, train_labels)
        self.classifier.load_state_dict(best_model)
        torch.save(self.classifier.state_dict(), os.path.join(save_dir,'best_classifier_crossvalidation.pth'))
        print('Classifier trained')
        self.classifier_trainer.evaluate_classifier(test_original_encoder_features_tensor, test_fused_features_tensor, test_labels)

    def train_and_evaluate_model_bootstrapped(self, newsfeeds_dir, save_dir='resources/models', n_bootstraps=10):
        all_newsfeeds, all_labels = self.load_data_from_dir(newsfeeds_dir=newsfeeds_dir)
        train_newsfeeds, test_newsfeeds, train_labels, test_labels = self.train_test_split(all_newsfeeds, all_labels)
        
        bootstrap_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        best_model = None
        best_f1_score = -float('inf')
        
        for i in range(n_bootstraps):
            print(f'Bootstrap iteration {i+1}/{n_bootstraps}')
            
            bootstrapped_newsfeeds, bootstrapped_labels = DataAugmentor.resample_with_replacement(texts=train_newsfeeds, labels=train_labels)
            
            print('Training AutoEncoder...')
            self.statistical_feature_extractor.train_autoencoder(train_newsfeeds=bootstrapped_newsfeeds)
            print('AutoEncoder trained')
            
            print('Evaluate Fused Features of bootstrapped feeds')
            train_fused_features_tensor, train_original_encoder_features_tensor = self.compute_fused_and_original_encoder_features(newsfeeds=bootstrapped_newsfeeds)
            print('Fused Features of bootstrapped feeds evaluated')
            
            print('Training Classifier...')
            metrics, best_f1_score_bootstrap, best_model_bootstrap = self.classifier_trainer.train_classifier_with_cross_validation(train_original_encoder_features_tensor, train_fused_features_tensor, bootstrapped_labels)
            
            if best_f1_score_bootstrap > best_f1_score and best_f1_score_bootstrap != 1:
                best_model = best_model_bootstrap

            for key in bootstrap_metrics:
                bootstrap_metrics[key].append(metrics[key])
            print('Classifier trained and evaluated for this bootstrap iteration')
        
        if best_model is not None:
            self.classifier.load_state_dict(best_model)
            torch.save(self.classifier.state_dict(), os.path.join(save_dir,'best_classifier_bootstrapped.pth'))
        
        print('Final evaluation metrics across all bootstrapped datasets:')
        for metric in bootstrap_metrics:
            mean_metric = np.mean(bootstrap_metrics[metric])
            std_metric = np.std(bootstrap_metrics[metric])
            print(f"{metric.capitalize()} - Mean: {mean_metric:.4f}, Std: {std_metric:.4f}")

        print('Evaluation on test dataset')
        test_fused_features_tensor, test_original_encoder_features_tensor = self.compute_fused_and_original_encoder_features(test_newsfeeds)
        self.classifier_trainer.evaluate_classifier(test_original_encoder_features_tensor, test_fused_features_tensor, test_labels)

    def detect_secret(self, newsfeed: list[str]) -> bool:
        self.statistical_feature_extractor.load_resources()
        model_path = 'resources/models/best_classifier_crossvalidation.pth'
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.classifier.load_state_dict(model_state_dict)
        self.classifier.eval()
        with torch.no_grad():
            semantic_features = self.semantic_feature_extractor(newsfeed)
            statistical_features = self.statistical_feature_extractor.get_statistical_features(newsfeed)
            
            if semantic_features.dim() == 2:  # [sequence_length, feature_size]
                semantic_features = semantic_features.unsqueeze(0)  # [1, sequence_length, feature_size]
            if statistical_features.dim() == 1:
                statistical_features = statistical_features.unsqueeze(0)  # [1, feature_size]

            fused_features = self.fusion_component(semantic_features, statistical_features)
            classifier_output = self.classifier(fused_features, semantic_features)
            probability_vector = torch.softmax(classifier_output, dim=1)
            predicted_label = torch.argmax(probability_vector, dim=1)

            decision = predicted_label.item() == 1

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