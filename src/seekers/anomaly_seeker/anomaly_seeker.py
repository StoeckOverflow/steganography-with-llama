from .features import *
from ..seeker import Seeker
from tqdm import tqdm
from ...utils.file_loading import create_dataset
from ...utils.string_modification import clean
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score
import os

class Anomaly_Seeker(Seeker):
    
    def __init__(self, disable_tqdm=False) -> None:
        super().__init__(disable_tqdm)

    def extract_features(self,articles):
        #llm = Llama(model_path='resources/llama-2-7b.Q5_K_M.gguf', logits_all=True, verbose=False)
        features = []
        for article in tqdm(articles, desc='Extracting features', disable=self.disable_tqdm):
            article = str(article)
            tokens = self.base_model.tokenizer().encode(article)

            # Article length
            length = len(article)

            # Average sentence length
            sentences = article.split('.')
            avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])

            # Type token ratio
            type_token_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

            # Flesch reading ease
            flesch_score = flesch_reading_ease(article)

            # Vocabulary richness
            vocab_richness = len(set(tokens)) / length if length > 0 else 0

            # Special characters count
            num_special_chars = special_chars_count(article)

            # Shannon entropy
            entropy = shannon_entropy(article)

            # Sentiment consistency
            sentiment = sentiment_consistency(article)

            # Use of named entities
            named_entities = named_entity_analysis(article)

            # Repetition patterns
            repetition = repetition_patterns(article)

            # Transition words
            #transition_words = count_transition_words(article)

            # Average token probability
            self.base_model.reset()
            self.base_model.eval(tokens)
            logits = np.array(self.base_model._scores)
            token_probabilities = softmax(logits)
            avg_token_probability = np.mean(token_probabilities)

            # Token probability variance
            #token_probability_variance = np.var(token_probabilities)

            # Collecting all features
            article_features = [length, avg_sentence_length, 
                                type_token_ratio, 
                                flesch_score, 
                                vocab_richness, 
                                num_special_chars, 
                                entropy, sentiment, 
                                named_entities, 
                                repetition, 
                                avg_token_probability,
                                ]
            
            features.append(article_features)

        scaler = StandardScaler()
        transformed_features = scaler.fit_transform(features)
        df = pd.DataFrame(transformed_features, columns=['length', 
                                                        'avg_sentence_length', 
                                                        'type_token_ratio', 
                                                        'flesch_score', 
                                                        'vocab_richness', 
                                                        'num_special_chars', 
                                                        'entropy', 
                                                        'sentiment', 
                                                        'named_entities', 
                                                        'repetition', 
                                                        'avg_token_probability',
                                                        ])
        return df

    def train_model(self,benign_data_path, malicious_data_path):
        print('Training model...')
        dataset = create_dataset(benign_data_path, malicious_data_path)
        articles = dataset['article']
        labels = dataset['label']
        features = self.extract_features(articles)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=420, stratify=labels)

        # Gridsearch with Cross Validation
        #param_grid = {'n_estimators': [100, 200, 300, 400, 500],
        #              'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        #clf = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5, scoring='f1')
        #clf.fit(X_train, y_train)
        #print(clf.best_params_)
        #print(clf.best_score_)
        #print(clf.best_estimator_)
        #print(clf.cv_results_)
        #print(clf.score(X_test, y_test))

        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=420)
        clf = clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        
        joblib.dump(clf, 'resources/models/anomaly_detector.joblib')
    
    def detect_secret(self, newsfeed: list[str]) -> bool:
        #print('Predicting news feed...')
        
        articles = [clean(article) for article in newsfeed]

        features = self.extract_features(articles)
        clf = joblib.load('resources/models/anomaly_detector.joblib')
        predictions = clf.predict(features)
        counts = Counter(predictions)
        #print('Predictions:', predictions)

        if counts[-1] >= 1:
            return True #-1
        else:
            return False #1

    def train_and_test_with_own_feeds(self):
        #benign_data_path = 'resources/feeds/clean_feeds.zip'
        #malicious_data_path = 'resources/doctored_feeds/doctored_feeds.zip'
        #train_model(benign_data_path, malicious_data_path)
        malicious_dir_path = 'resources/doctored_feeds/'
        for feed in os.listdir(malicious_dir_path):
            if feed.endswith('.zip'):
                continue
            feed_path = os.path.join(malicious_dir_path, feed)
            result = self.predict_single_feed(feed_path)
            true_label = -1
            print(f"Prediction for {feed}: {result} ({result == true_label})")

        benign_dir_path = 'resources/feeds/'
        for feed in os.listdir(benign_dir_path):
            if feed.endswith('.zip'):
                continue
            feed_path = os.path.join(benign_dir_path, feed)
            result = self.predict_single_feed(feed_path)
            true_label = 1
            print(f"Prediction for {feed}: {result} ({result == true_label})")