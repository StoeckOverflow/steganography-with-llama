from .features import *
from ..seeker import Seeker
from tqdm import tqdm
from ...utils.string_modification import clean
from ...utils.llama_utils import get_probabilities, get_entropy, get_perplexity
import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import json
import os

class Anomaly_Seeker(Seeker):

    def __init__(self, disable_tqdm=False) -> None:
        super().__init__(disable_tqdm)
        perplexity_statistics = pd.read_csv('resources/perplexity_statistics.csv')
        self.mean_perplexity = perplexity_statistics[perplexity_statistics['Statistic'] == 'Mean Perplexity']['Value'].iloc[0]
        self.std_perplexity = perplexity_statistics[perplexity_statistics['Statistic'] == 'Standard Deviation']['Value'].iloc[0]
        self.baseline_feature_set = pd.read_csv('resources/baseline_features.csv')

    def extract_features_in_articles(self, articles):
        features = []
        for article in articles:
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
            #entropy = shannon_entropy(article)
            sentiment = sentiment_consistency(article)
            named_entities = named_entity_analysis(article)
            repetition = repetition_patterns(article)
            
            # Llama Features
            probs = get_probabilities(self.base_model, tokens, article)
            avg_token_probability = np.mean(probs)
            
            # Reflects the diversity or uncertainty of token predictions.
            # Tokens with low log-probability are less suspicious in high-entropy (uniform) distributions
            # High Entropy: Indicates uncertainty, many tokens are equally likely
            # Low Entropy: (Spiky Distribution): Indicates certainty, fewer tokens are likely
            # 1 Compute the entropy of the token distribution at each position in the text.
            # 2 Assess the log-probability of the actual token at each position.
            # 3 Decision Metric: A low log-probability token in a low-entropy context indicates a potential anomaly
            entropy = get_entropy(probs)
            
            # Perplexity measures a model’s uncertainty in predicting the next token (Ranges: 1 to inf)
            # Lower values indicate a more predictable and effective model
            perplexity = get_perplexity(probs, tokens)
            perplexity_scaled = (perplexity - self.std_perplexity) / self.mean_perplexity

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
        '''
        columns_to_scale = ['length', 'avg_sentence_length','type_token_ratio', 'flesch_score', 'vocab_richness','num_special_chars', 'entropy', 
                            'sentiment', 'named_entities', 'repetition', 'avg_token_probability'] 
        columns_not_to_scale = [col for col in features.columns if col not in columns_to_scale]
        '''
        features_array = np.array(features)
        transformed_features = scaler.fit_transform(features_array[:, :-1])
        transformed_features = np.column_stack((transformed_features, features_array[:, -1]))
        
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

    def extract_features_and_labels_in_articles(self, newsfeeds_directory_path, save_flag=True):
        print('Start Feature Extraction...')
        newsfeeds_files_pattern = os.path.join(newsfeeds_directory_path,'*.json')
        all_labels = []
        feed_paths = glob.glob(newsfeeds_files_pattern)
        i = 0
        for feed_path in tqdm(feed_paths, desc='Extract Features of Newsfeeds', disable=self.disable_tqdm):
            with open(feed_path, 'r') as file:
                parsed_feed = json.load(file)
            feed_array = parsed_feed['feed']
            feed_labels = parsed_feed['labels']
            new_features = self.extract_features_in_articles(feed_array)
            if i == 0:
                feature_set = new_features
                i += 1
            else:
                feature_set = pd.concat([feature_set, new_features], ignore_index=True)
            all_labels.extend(feed_labels)
        
        feature_set['label'] = all_labels
        
        if save_flag:
            print('Save feature_set to csv...')
            feature_set.to_csv('resources/feature_set_articles.csv', index=False)
            print('feature-set saved')
        
        return feature_set
    
    def extract_features_in_newsfeed(self, newsfeed):

        article_features = []
        for article in tqdm(newsfeed, desc='Extracting features', disable=self.disable_tqdm):
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
            sentiment = sentiment_consistency(article)
            named_entities = named_entity_analysis(article)
            repetition = repetition_patterns(article)
            
            # Llama Features
            probs = get_probabilities(self.base_model, tokens, article)
            avg_token_probability = np.mean(probs)
            
            # Reflects the diversity or uncertainty of token predictions.
            # Tokens with low log-probability are less suspicious in high-entropy (uniform) distributions
            # High Entropy: Indicates uncertainty, many tokens are equally likely
            # Low Entropy: (Spiky Distribution): Indicates certainty, fewer tokens are likely
            # 1 Compute the entropy of the token distribution at each position in the text.
            # 2 Assess the log-probability of the actual token at each position.
            # 3 Decision Metric: A low log-probability token in a low-entropy context indicates a potential anomaly
            entropy = get_entropy(probs)
            
            # Perplexity measures a model’s uncertainty in predicting the next token (Ranges: 1 to inf)
            # Lower values indicate a more predictable and effective model
            perplexity = get_perplexity(probs, tokens)

            article_features_entry = [length, 
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
            
            article_features.append(article_features_entry)

        feature_set = pd.DataFrame(article_features, columns=[
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
                                                            'perplexity_scaled'])
        
        result_frame = pd.DataFrame(columns=['length_ks_statistic', 'length_ad_statistic', 'length_t_statistic', 'avg_sentence_length_ks_statistic', 'avg_sentence_length_ad_statistic', 'avg_sentence_length_t_statistic', 'type_token_ratio_ks_statistic', 'type_token_ratio_ad_statistic', 'type_token_ratio_t_statistic', 'flesch_score_ks_statistic', 'flesch_score_ad_statistic', 'flesch_score_t_statistic', 'vocab_richness_ks_statistic', 'vocab_richness_ad_statistic', 'vocab_richness_t_statistic', 'num_special_chars_ks_statistic', 'num_special_chars_ad_statistic', 'num_special_chars_t_statistic', 'entropy_ks_statistic', 'entropy_ad_statistic', 'entropy_t_statistic', 'sentiment_ks_statistic', 'sentiment_ad_statistic', 'sentiment_t_statistic', 'named_entities_ks_statistic', 'named_entities_ad_statistic', 'named_entities_t_statistic', 'repetition_ks_statistic', 'repetition_ad_statistic', 'repetition_t_statistic', 'avg_token_probability_ks_statistic', 'avg_token_probability_ad_statistic', 'avg_token_probability_t_statistic', 'perplexity_scaled_ks_statistic', 'perplexity_scaled_ad_statistic', 'perplexity_scaled_t_statistic'])
        
        for col in tqdm(feature_set.columns, desc='Estimate statistics for features', disable=self.disable_tqdm):
            result_frame.at[0, f"{col}_ks_statistic"] = perplexity_ks_test(self.baseline_feature_set[col], feature_set[col])
            result_frame.at[0, f"{col}_ad_statistic"] = perplexity_ad_test(self.baseline_feature_set[col], feature_set[col])
            result_frame.at[0, f"{col}_t_statistic"] = perplexity_t_test(feature_set[col])
        
        return result_frame

    def extract_features_and_labels_in_newsfeeds(self, newsfeeds_directory_path, save_flag=True):
        print('Start Feature Extraction...')        
        
        benign = os.path.join(newsfeeds_directory_path, '*;1')
        malicious = os.path.join(newsfeeds_directory_path, '*;-1')

        feed_paths_benign = glob.glob(benign)
        feed_paths_malicious = glob.glob(malicious)
        
        feed_paths = feed_paths_benign + feed_paths_malicious
        
        i = 0
        feature_set = pd.DataFrame(columns=['length_ks_statistic', 'length_ad_statistic', 'length_t_statistic', 'avg_sentence_length_ks_statistic', 'avg_sentence_length_ad_statistic', 'avg_sentence_length_t_statistic', 'type_token_ratio_ks_statistic', 'type_token_ratio_ad_statistic', 'type_token_ratio_t_statistic', 'flesch_score_ks_statistic', 'flesch_score_ad_statistic', 'flesch_score_t_statistic', 'vocab_richness_ks_statistic', 'vocab_richness_ad_statistic', 'vocab_richness_t_statistic', 'num_special_chars_ks_statistic', 'num_special_chars_ad_statistic', 'num_special_chars_t_statistic', 'entropy_ks_statistic', 'entropy_ad_statistic', 'entropy_t_statistic', 'sentiment_ks_statistic', 'sentiment_ad_statistic', 'sentiment_t_statistic', 'named_entities_ks_statistic', 'named_entities_ad_statistic', 'named_entities_t_statistic', 'repetition_ks_statistic', 'repetition_ad_statistic', 'repetition_t_statistic', 'avg_token_probability_ks_statistic', 'avg_token_probability_ad_statistic', 'avg_token_probability_t_statistic', 'perplexity_scaled_ks_statistic', 'perplexity_scaled_ad_statistic', 'perplexity_scaled_t_statistic'])
        for feed_path in feed_paths:
            with open(feed_path, 'r') as file:
                parsed_feed = json.load(file)
            feed_array = parsed_feed['feed']
            new_features_frame = self.extract_features_in_newsfeed(feed_array)
            new_features_frame['label'] = feed_path.split(';')[1]
            if i == 0:
                feature_set = new_features_frame
                i += 1
            else:
                feature_set = pd.concat([feature_set, new_features_frame], ignore_index=True)
        if save_flag:
            print('Save feature_set to csv...')
            feature_set.to_csv('resources/feature_set_newsfeeds.csv', index=False)
            print('feature-set saved')
        
        return feature_set
       
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
    
    def train_model(self, modelName='RFC', permutation_importance_flag=True, plotting_flag=True):
        print('Preparation...')
        try:
            feature_set = pd.read_csv('resources/feature_set_newsfeeds.csv')
        except FileNotFoundError:
            feature_set = self.extract_features_and_labels_in_newsfeeds('resources/feeds/doctored_feeds_newsfeeds')
        
        labels = feature_set.label
        feature_set = feature_set.drop(columns=['label'])
        X_train, X_test, y_train, y_test = train_test_split(feature_set, labels, test_size=0.2, random_state=420, stratify=labels)
        
        if modelName == 'RFC':
            print('Training Random Forest Classifier')
            clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=420)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
        elif modelName == 'SVM':
            clf_svm = SVC(random_state=420)
            clf_svm.fit(X_train, y_train)
            y_pred = clf_svm.predict(X_test)
            
        elif (modelName == 'CoM'):
            print("Training Center of Mass model")
            threshold_factor=0.5 #Param do be adjusted
            centroid = np.mean(X_test, axis=0)
            distances = np.sqrt(np.sum((X_test - centroid)**2, axis=1))
            threshold = threshold_factor * np.std(distances)
            y_pred = distances > threshold
            
        elif modelName == 'CoN':
            print("Training Center of Neighborhood model")
            k = 5
            threshold_factor = 1.5
            neighbors = NearestNeighbors(n_neighbors=k + 1)  # +1 because a point is its own nearest neighbor
            neighbors.fit(X_test)
            distances, indices = neighbors.kneighbors(X_test)

            neighborhood_centroids = np.mean(X_test.iloc[indices][:, 1:, :], axis=1)  # Exclude the point itself
            point_to_neighborhood_distances = np.linalg.norm(X_test - neighborhood_centroids, axis=1)
            threshold = threshold_factor * np.std(point_to_neighborhood_distances)
            y_pred = point_to_neighborhood_distances > threshold
        
        print('\n')
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print('\n')
        
        if permutation_importance_flag:
            print('Calculate Permutation Importance')
            feature_importance_results = permutation_importance(clf, X_test, y_pred, scoring='accuracy')
            importances = feature_importance_results.importances_mean
            
            # Plot the Permutation Importance
            feature_names = np.array(feature_set.columns)
            plt.figure(figsize=(20, 8))
            plt.barh(feature_names, importances)
            plt.xlabel('Mean decrease in accuracy')
            plt.ylabel('Feature')
            plt.title('Feature importance')
            plt.savefig(f"permutation_importance_{modelName}_newsfeeds.png")
        
        if plotting_flag:
            print('Plot Prediction')
            x_test_dataframe = pd.DataFrame(X_test)
            x_test_dataframe['prediction'] = y_pred
            x_test_dataframe = x_test_dataframe.reset_index(drop=True)
            self.plot_predictions(x_test_dataframe, modelName)
        
        joblib.dump(clf, 'resources/models/anomaly_detector_newsfeeds.joblib')
    
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
       
    def plot_predictions(self, df, modelname='RFC'):
        predictions = df.prediction
        features = df.drop(columns=['prediction'])
        
        principle_components = self.apply_PCA(df)
        df_pca = pd.DataFrame(principle_components, columns=['x', 'y'])
        df_pca['prediction'] = predictions
        df_pca.plot.scatter(x='x', y='y', c='prediction', colormap='viridis')
        
        plt.savefig(f"predictions_{modelname}_newsfeeds.png")
        
        outliers = df_pca[df_pca['prediction'] == -1]
        outlier_data = outliers[['x', 'y']]
        print(f"Number of Outliers: {len(outlier_data)}")

        outlier_data.to_csv(f"outliers_{modelname}_newsfeeds.csv", index=False)
        
    def apply_PCA(self, df, n_components=2):
            pca = PCA(n_components=n_components)
            pca.fit(df)
            return pca.transform(df)