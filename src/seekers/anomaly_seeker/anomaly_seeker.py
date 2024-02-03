import os
import json
import glob
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

from .features import *
from ..seeker import Seeker
from ...utils.string_modification import clean
from ...utils.llama_utils import get_entropy, get_perplexity
from ...utils.bert_utils import get_bert_next_token_probability_and_logits

class Anomaly_Seeker(Seeker):

    def __init__(self, disable_tqdm=False) -> None:
        super().__init__(disable_tqdm)
        self.base_model = self.bert_model # Use BERT as base model instead of LLAMA
        self.scoring = 'f1' # Optimize for F1-score
        self.random_state = 420

    def extract_features_in_newsfeed(self, newsfeed):
        feature_sets = []

        for article in tqdm(newsfeed, desc='Extracting Features', disable=self.disable_tqdm):
            article = str(article)
            article = clean(article)

            article_length = len(article)
            num_special_chars = special_chars_count(article)
            sentences = article.split('. ')
            avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])
            flesch_score = flesch_reading_ease(article)
            sentiment = TextBlob(article).sentiment.polarity
            true_token_probs, softmax_logits = get_bert_next_token_probability_and_logits(self.base_model, self.bert_tokenizer, article, initial_context_length=5)
            avg_token_probability = np.mean(true_token_probs)
            entropy = get_entropy(softmax_logits)
            perplexity = get_perplexity(self.bert_model, self.bert_tokenizer, article)

            feature_sets.append([article_length, avg_sentence_length, flesch_score, sentiment, num_special_chars, entropy, avg_token_probability, perplexity])

        # Convert to NumPy array for easier calculations
        feature_array = np.array(feature_sets)

        # Calculate different aggregations
        mean_features = np.mean(feature_array, axis=0)
        var_features = np.var(feature_array, axis=0)
        min_features = np.min(feature_array, axis=0)
        max_features = np.max(feature_array, axis=0)
        std_features = np.std(feature_array, axis=0)

        # Combine all features into one DataFrame
        columns = ['article_length', 'sentence_length', 'flesch_score', 'sentiment', 'num_special_chars', 'entropy', 'token_probability', 'perplexity']
        aggregated_features = np.concatenate([mean_features, var_features, min_features, max_features, std_features])
        aggregated_columns = [f"{agg}_{col}" for agg in ['mean', 'var', 'min', 'max', 'std'] for col in columns]
        feature_set = pd.DataFrame([aggregated_features], columns=aggregated_columns)
        
        return feature_set


    def extract_features_and_labels_in_newsfeeds(self, newsfeeds_directory_path, save_flag=True):
        print('Start Feature Extraction...')        

        benign = os.path.join(newsfeeds_directory_path, '*;1')
        malicious = os.path.join(newsfeeds_directory_path, '*;-1')

        feed_paths_benign = glob.glob(benign)
        feed_paths_malicious = glob.glob(malicious)
        feed_paths = feed_paths_benign + feed_paths_malicious
        random.shuffle(feed_paths)
        
        feature_set = pd.DataFrame()
        for i, feed_path in enumerate(feed_paths):
            print(f"Feed Path: {feed_path}\nIteration: {i}")
            with open(feed_path, 'r') as file:
                parsed_feed = json.load(file)
            feed_array = parsed_feed['feed']

            new_features_frame = self.extract_features_in_newsfeed(feed_array)
            new_features_frame['label'] = int(feed_path.split(';')[-1])  # Extracting label from filename

            feature_set = pd.concat([feature_set, new_features_frame], ignore_index=True)
            if save_flag:
                feature_set.to_csv('resources/feature_set_newsfeeds.csv', index=False)
            print(f"Feed with feed_path {feed_path} saved\nNext Iteration: {i+1}")

        print('Feature extraction completed.')
        if save_flag:
            print('Feature-set saved to csv.')

        return feature_set


    def train_and_evaluate(self, X_train, y_train, X_test, y_test, model_name, param_dist):
        print(f'Training {model_name} with RandomizedSearchCV using Stratified Cross Validation')
        if model_name == 'RFC':
            clf = RandomForestClassifier(random_state=self.random_state)
        elif model_name == 'SVM':
            clf = SVC(random_state=self.random_state)
        elif model_name == 'OCSVM':
            # Split validation set from training set
            X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train)
            # Only use the normal data for training
            indices = np.where(y_train == 1)[0] 
            X_train = X_train[indices]
            
            best_params = self.manual_gridsearch(X_train, X_validation, y_validation, param_dist)
            clf = OneClassSVM(**best_params)
            clf.fit(X_train)
            y_pred = clf.predict(X_test)

            # Plot ROC curve and find optimal threshold
            optimal_threshold = self.plot_roc_curve(clf, X_test, y_test, model_name)
            print(f"{optimal_threshold=}")
            # Apply the optimal threshold to the decision function scores
            decision_scores = clf.decision_function(X_test)
            decision_scores = clf.decision_function(X_test)
            y_pred_optimal = np.where(decision_scores < optimal_threshold, -1, 1) # -1 for anomalies, 1 for normal data
            print(np.unique(y_pred_optimal, return_counts=True))
            # Print metrics
            self.print_metrics(y_test, y_pred_optimal)

            return clf, optimal_threshold

        # Set up Stratified Cross Validation
        stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100, cv=stratified_cv, verbose=2, random_state=self.random_state, n_jobs=-1, scoring=self.scoring)
        random_search.fit(X_train, y_train)
        print("Best parameters found: ", random_search.best_params_)
        clf = random_search.best_estimator_
        y_pred = clf.predict(X_test)

        self.print_metrics(y_test, y_pred)
        return clf
    

    def manual_gridsearch(self, X_train, X_validation, y_validation, param_dist):
        best_params = None
        best_score = -np.inf
        for params in ParameterGrid(param_dist):
            clf = OneClassSVM(**params)
            clf.fit(X_train)
            y_pred = clf.predict(X_validation)
            score = f1_score(y_validation, y_pred, pos_label=-1)
            if score > best_score:
                best_score = score
                best_params = params
        print("Best parameters found: ", best_params)
        return best_params
    

    def print_metrics(self, y_test, y_pred):
        print(f"Precision: {precision_score(y_test, y_pred, pos_label=-1)}")
        print(f"Recall: {recall_score(y_test, y_pred, pos_label=-1)}")
        print(f"F1-score: {f1_score(y_test, y_pred, pos_label=-1)}")
        print(f"Accuracy: {np.mean(y_test == y_pred)}")


    def load_or_extract_features(self, features_file, feeds_directory):
        try:
            feature_set = pd.read_csv(features_file)
            print("Features loaded from file.")
        except FileNotFoundError:
            print("Features file not found. Extracting features...")
            feature_set = self.extract_features_and_labels_in_newsfeeds(feeds_directory)
        labels = [int(label) for label in feature_set.label]
        feature_set = feature_set.drop(columns=['label'])
        return feature_set, labels


    def train_model(self, modelName='SVM'):
        print('Preparation...')
        feature_set, labels = self.load_or_extract_features('resources/feature_set_newsfeeds.csv', 'resources/feeds/kaggle')

        # Initial Training
        clf, best_threshold, X_test, y_test, column_names = self.perform_training(feature_set, labels, modelName)

        self.calculate_and_plot_permutation_importance(clf, X_test, y_test, column_names, modelName)

        # Save the final model
        joblib.dump(clf, f"resources/models/anomaly_detector_{modelName}.joblib")
        joblib.dump(best_threshold, f"resources/models/best_threshold_{modelName}.joblib")


    def perform_training(self, feature_set, labels, modelName):
        X_train, X_test, y_train, y_test, column_names = self.prepare_data(feature_set, labels)
        clf, best_threshold = self.train_and_evaluate(X_train, y_train, X_test, y_test, modelName, self.get_param_dist(modelName))
        return clf, best_threshold, X_test, y_test, column_names
    

    def prepare_data(self, feature_set, labels):
        feature_set = feature_set.fillna(0.0)

        X_train, X_test, y_train, y_test = train_test_split(feature_set, labels, test_size=0.2, random_state=self.random_state, stratify=labels)
        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

        # Using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Selecting top features using anova f-test
        selector = SelectKBest(f_classif, k=3)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        # Save scaler and selector for later use
        joblib.dump(scaler, 'resources/models/scaler.joblib')
        joblib.dump(selector, 'resources/models/selector.joblib')

        selected_columns_names = feature_set.columns[selector.get_support()]
        
        return X_train, X_test, y_train, y_test, selected_columns_names
    

    def calculate_and_plot_permutation_importance(self, clf, X_test, y_test, column_names, modelName):
        importances = permutation_importance(clf, X_test, y_test, scoring=self.scoring)
        self.plot_importance(importances, column_names, modelName)
        print('Permutation Importance Plot saved to file.')


    def plot_importance(self, importances, feature_names, modelName):
        # Calculate the mean decrease for each feature
        mean_importances = np.mean(importances.importances, axis=1)
        
        # Sort the feature names based on the mean decrease
        sorted_indices = np.argsort(mean_importances)
        sorted_feature_names = [feature_names[idx] for idx in sorted_indices]
        sorted_importances = importances.importances[sorted_indices]
        
        plt.figure(figsize=(20, 8))
        box = plt.boxplot(sorted_importances.T, vert=False, labels=sorted_feature_names, patch_artist=True)

        # Customizing the colors
        for patch, color in zip(box['boxes'], ['skyblue', 'lightgreen', 'tan', 'pink', 'lightyellow'] * 2):
            patch.set_facecolor(color)

        # Increasing label sizes
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)

        # Setting up the gridlines
        plt.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)

        # Customizing the median lines
        for median in box['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        # Customizing the fliers (outliers)
        for flier in box['fliers']:
            flier.set_marker('o')
            flier.set_color('orange')
            flier.set_alpha(0.5)

        # Adjust axis labels and title
        plt.xlabel(f'Decrease in {self.scoring} score', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title(f'Feature Importance Boxplot for {modelName}', fontsize=16)

        # Saving the figure with high quality
        plt.tight_layout()
        plt.savefig(f"permutation_importance_boxplot_{modelName}.png", dpi=300)

    
    def plot_roc_curve(self, clf, X_test, y_test, modelName):
        decision_scores = clf.decision_function(X_test)
        #y_test = np.where(y_test == 1, 0, 1) # Convert labels to 0 and 1 as required by roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, decision_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_{modelName}.png', dpi=300)

        # Find the optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold


    def get_param_dist(self, modelName):
        if modelName == 'RFC':
            return {
                'n_estimators': [80, 100, 120],  # Around 100
                'max_depth': [3, 4, 5],  # Around 4
                'max_features': ['sqrt', 'log2'],  # 'sqrt' was best, but still worth checking 'log2'
                'min_samples_split': [8, 10, 12],  # Around 10
                'min_samples_leaf': [3, 4, 5],  # Around 4
                'bootstrap': [True, False]  
            }
            
        elif modelName == 'SVM':
            return {
                'C': [0.1, 1, 10, 100],  # Regularization parameter
                'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],  # Kernel coefficient
                'kernel': ['rbf', 'poly', 'sigmoid'],  # Kernel type
                'degree': [2, 3, 4]  # Degree for 'poly' kernel. Consider lower values to avoid overfitting.
            }    
        
        elif modelName == 'OCSVM':
            return {
                'nu': [0.01, 0.05, 0.1], 
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        
        else:
            raise ValueError(f"Model name {modelName} not recognized.")

    def detect_secret(self, newsfeed: list[str]) -> bool:
        # Extract features from the newsfeed
        features = self.extract_features_in_newsfeed(newsfeed)

        # Scale and select features
        scaler = joblib.load('resources/models/scaler.joblib')
        selector = joblib.load('resources/models/selector.joblib')
        features_scaled = scaler.transform(features)
        features_selected = selector.transform(features_scaled)

        # Load the pretrained classifier and predict if a secret is present
        clf = joblib.load('resources/models/anomaly_detector_OCSVM.joblib')
        threshold = joblib.load('resources/models/best_threshold_OCSVM.joblib')
        
        # Use the decision function to get the score
        decision_score = clf.decision_function(features_selected)

        # Apply the threshold to the decision score to determine the prediction
        prediction = (decision_score < threshold).astype(int) # Returns 1 for anomalies and 0 for normal data
        
        # Return True if the model predicts the presence of a secret, False otherwise
        return bool(prediction)