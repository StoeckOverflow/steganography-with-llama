from .features import *
from ..seeker import Seeker
from tqdm import tqdm
from ...utils.file_loading import create_dataset
from ...utils.string_modification import clean
from ...utils.llama_utils import get_probabilities, get_entropy, get_perplexity
import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import json

class Anomaly_Seeker(Seeker):
    
    def __init__(self, disable_tqdm=False) -> None:
        super().__init__(disable_tqdm)
        perplexity_statistics = pd.read_csv('resources/perplexity_statistics.csv')
        self.mean_perplexity = perplexity_statistics[perplexity_statistics['Statistic'] == 'Mean Perplexity']['Value'].iloc[0]
        self.std_perplexity = perplexity_statistics[perplexity_statistics['Statistic'] == 'Standard Deviation']['Value'].iloc[0]

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
            probs = get_probabilities(self.base_model, article)
            avg_token_probability = np.mean(probs)
            
            # Reflects the diversity or uncertainty of token predictions.
            # Tokens with low log-probability are less suspicious in high-entropy (uniform) distributions
            # High Entropy: Indicates uncertainty, many tokens are equally likely
            # Low Entropy: (Spiky Distribution): Indicates certainty, fewer tokens are likely
            # 1 Compute the entropy of the token distribution at each position in the text.
            # 2 Assess the log-probability of the actual token at each position.
            # 3 Decision Metric: A low log-probability token in a low-entropy context indicates a potential anomaly
            entropy = get_entropy(self.base_model, article)
            
            # Perplexity measures a model’s uncertainty in predicting the next token (Ranges: 1 to inf)
            # Lower values indicate a more predictable and effective model
            perplexity_scaled = (get_perplexity(self.base_model, article) - self.std_perplexity) / self.mean_perplexity

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
                                perplexity_scaled
                                ]
            
            features.append(article_features)
        
        scaler = StandardScaler()
        transformed_features = scaler.fit_transform(features)
        df = pd.DataFrame(transformed_features, columns=[
                                                        'length', 
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
                                                        'perplexity_scaled'
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
    
    def train_model(self, permutation_importance_flag=True, plotting_flag=True):
        print('Training model...')
        feature_set = pd.DataFrame(columns=['length', 
                                            'avg_sentence_length', 
                                            'type_token_ratio', 
                                            'flesch_score', 
                                            'vocab_richness', 
                                            'num_special_chars', 
                                            'entropy', 
                                            'sentiment', 
                                            'named_entities', 
                                            'repetition', 
                                            'avg_token_probability'])
        labels = []
        news_feeds_directory_path = 'resources/feeds/doctored_feeds_new/*.json'
        feed_paths = glob.glob(news_feeds_directory_path)
        for feed_path in tqdm(feed_paths, desc='Features extracted of feed paths'):
            with open(feed_path, 'r') as file:
                parsed_feed = json.load(file)
            feed_array = parsed_feed['feed']
            feed_labels = parsed_feed['labels']
            feature_set = pd.concat([feature_set, self.extract_features(feed_array)], ignore_index=True)
            labels.append(feed_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(feature_set, labels, test_size=0.2, random_state=420, stratify=labels)
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=420)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        
        if permutation_importance_flag:
            feature_importance_results = permutation_importance(clf, feature_set, y_pred, scoring='accuracy')
            importances = feature_importance_results.importances_mean
            
            # Plot the Permutation Importance
            feature_names = np.array(feature_set.columns)
            plt.figure(figsize=(20, 8))
            plt.barh(feature_names, importances)
            plt.xlabel('Mean decrease in accuracy')
            plt.ylabel('Feature')
            plt.title('Feature importance')
            plt.savefig(f"permutation_importance_RandomForest.png")
        
        if plotting_flag:
            feature_set['predictions'] = y_pred
            self.plot_predictions(feature_set)
        
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
       
    def plot_predictions(df, modelname='RandomForest'):
        predictions = df.predictions
        features = df.drop(columns=['predictions'])
        
        def apply_PCA(df, n_components=2):
            pca = PCA(n_components=n_components)
            pca.fit(df)
            return pca.transform(df)

        principle_components = apply_PCA()
        df_pca = pd.DataFrame(principle_components, columns=['x', 'y'])
        df_pca['predictions'] = predictions

        df_pca.plot.scatter(x='x', y='y', c='prediction', colormap='viridis')
        plt.savefig(f"predictions_{modelname}.png")

        outliers = df_pca[df_pca['prediction'] == -1]
        outlier_data = outliers[['x', 'y']]
        print(f"Number of Outliers: {len(outlier_data)}")

        outlier_data.to_csv(f"outliers_{modelname}.csv", index=False)