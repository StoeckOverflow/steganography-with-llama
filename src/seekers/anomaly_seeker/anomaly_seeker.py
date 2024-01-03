from .features import *
from ..seeker import Seeker
from tqdm import tqdm
from ...utils.file_loading import create_dataset
from ...utils.string_modification import clean
from ...utils.llama_utils import get_probabilities, get_entropy, get_perplexity
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score
from sklearn.inspection import permutation_importance
import os
from sklearn.decomposition import PCA

class Anomaly_Seeker(Seeker):
    
    def __init__(self, disable_tqdm=False) -> None:
        super().__init__(disable_tqdm)

    def extract_features(self,articles):
        features = []
        for article in tqdm(articles, desc='Extracting features', disable=self.disable_tqdm):
            article = str(article)
            
            # Standard Text Features
            tokens = self.base_model.tokenizer().encode(article)
            length = len(article)
            sentences = article.split('.')
            avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])
            type_token_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
            flesch_score = flesch_reading_ease(article)
            vocab_richness = len(set(tokens)) / length if length > 0 else 0
            num_special_chars = special_chars_count(article)
            entropy = shannon_entropy(article)
            sentiment = sentiment_consistency(article)
            named_entities = named_entity_analysis(article)
            repetition = repetition_patterns(article)
            
            # Llama Features
            probs = get_probabilities(self.llm, article)
            log_probs = np.log(probs)
            avg_token_probability = np.mean(probs)
            
            # Perplexity measures a model’s uncertainty in predicting the next token
            # Lower values indicate a more predictable and effective model
            # Ranges: 1 to inf
            
            # Analyze 5 newsfeeds, each with 30 articles, to establish a baseline of perplexity scores.
            # Compute perplexity for each article to gauge the "normal" range.
            # Anomaly Detection: Compare new articles’ perplexity against this baseline.
            # To Do: Z-Score normalization of article perplexities
            
            # Goal: Identify if article has high z-score
            # Important: Baseline Calculation and integration of comparison against baseline
            perplexity = get_perplexity(self.llm, article)
            
            # Reflects the diversity or uncertainty of token predictions.
            # Tokens with low log-probability are less suspicious in high-entropy (uniform) distributions
            # High Entropy: Indicates uncertainty, many tokens are equally likely
            # Low Entropy (Spiky Distribution): Indicates certainty, fewer tokens are likely
            # 1 Compute the entropy of the token distribution at each position in the text.
            # 2 Assess the log-probability of the actual token at each position.
            # 3 Decision Metric: A low log-probability token in a low-entropy context indicates a potential anomaly
            entropy = get_entropy(self.llm, article, probs,log_probs)

            article_features = [length, 
                                avg_sentence_length, 
                                type_token_ratio, 
                                flesch_score, 
                                vocab_richness, 
                                num_special_chars, 
                                entropy, 
                                sentiment, 
                                named_entities, 
                                repetition, 
                                avg_token_probability,
                                perplexity
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

    def gridSearch(self, features, labels):
        'Gridsearch with Cross Validation'

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=420, stratify=labels)
        param_grid = {'n_estimators': [100, 200, 300, 400, 500],
                          'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        clf = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5, scoring='f1')
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        print(clf.best_score_)
        print(clf.best_estimator_)
        print(clf.cv_results_)
        print(clf.score(X_test, y_test))
    
    def train_model(self,benign_data_path, malicious_data_path, permutation_importance_flag=True):
        print('Training model...')
        
        dataset = create_dataset(benign_data_path, malicious_data_path)
        articles = dataset['article']
        labels = dataset['label']
        
        features = self.extract_features(articles)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=420, stratify=labels)
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=420)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        
        if permutation_importance_flag:
            feature_importance_results = permutation_importance(clf, features, y_pred, scoring='accuracy')
            importances = feature_importance_results.importances_mean
            
            # Plot the Permutation Importance
            feature_names = np.array(features.columns)

            # Plotting
            plt.figure(figsize=(20, 8))
            plt.barh(feature_names, importances)
            plt.xlabel('Mean decrease in accuracy')
            plt.ylabel('Feature')
            plt.title('Feature importance')
            plt.savefig(f"permutation_importance_RandomForest.png")
        
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
       
    def plot_predictions(df, ftp_traffic=None, modelname='IsolationForest'):
        
        def apply_PCA(features, n_components=2):
            pca = PCA(n_components=n_components)
            pca.fit(features)
            return pca.transform(features)
        
        predictions = df.prediction
        features = df.drop(columns=['prediction', 'text_num'])

        principle_components = apply_PCA(features)
        df_pca = pd.DataFrame(principle_components, columns=['x', 'y'])
        df_pca['prediction'] = predictions
        df_pca['text_num'] = df['text_num']

        df_pca.plot.scatter(x='x', y='y', c='prediction', colormap='viridis')
        plt.savefig(f"predictions_{modelname}.png")

        outliers = df_pca[df_pca['prediction'] == 0]
        outlier_data = outliers[['text_num', 'x', 'y']]
        print(f"Number of Outliers: {len(outlier_data)}")
        if ftp_traffic is not None:
            outlier_data = outlier_data.merge(ftp_traffic, on='text_num', how='left')
        outlier_data.to_csv(f"outliers_{modelname}.csv", index=False)
