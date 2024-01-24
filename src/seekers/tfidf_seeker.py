import pandas as pd
from ..utils.string_modification import clean
from .seeker import Seeker
import json
import os
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

class Anomaly_Seeker_TFIDF(Seeker):
    def __init__(self, disable_tqdm=False) -> None:
        super().__init__(disable_tqdm)

    def extract_tfidf_features(self, feed_path):
        feeds = pd.DataFrame(columns=['feed', 'label'])

        # extract features from doctored feeds
        doctored_feeds = []
        for file in os.listdir(os.path.join(feed_path, 'doctored_feeds')):
            feed = json.load(open(os.path.join(feed_path, 'doctored_feeds', file), 'r'))
            article_array = feed['feed']
            article_array = [clean(article) for article in article_array]
            feed_string = ' '.join(article_array)
            doctored_feeds.append(feed_string)
        doctored_feeds_df = pd.DataFrame({'feed': doctored_feeds, 'label': -1})
        
        # extract features from clean feeds
        clean_feeds = []
        for file in os.listdir(os.path.join(feed_path, 'clean_feeds')):
            feed = json.load(open(os.path.join(feed_path, 'clean_feeds', file), 'r'))
            article_array = feed['feed']
            article_array = [clean(article) for article in article_array]
            feed_string = ' '.join(article_array)
            clean_feeds.append(feed_string)
        clean_feeds_df = pd.DataFrame({'feed': clean_feeds, 'label': 1})
        
        feeds_df = pd.concat([doctored_feeds_df, clean_feeds_df], ignore_index=True)
        feeds = feeds_df.feed
        labels = feeds_df.label

        vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_df=0.8, min_df=10, norm='l2', analyzer='word')
        features = vectorizer.fit_transform(feeds)
        features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())
        joblib.dump(vectorizer, 'resources/models/tfidf_vectorizer.joblib')

        features_df['label'] = labels

        features_df.to_csv('resources/tfidf_features.csv', index=False)

        return features_df

    def train_model(self):
        try:
            features_df = pd.read_csv('resources/tfidf_features.csv')
        except FileNotFoundError:
            features_df = self.extract_tfidf_features('resources/feeds/')

        labels = features_df.label
        features = features_df.drop(columns=['label'])
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=420, stratify=labels)
        
        clf = IsolationForest(random_state=420, n_estimators=100, max_samples=0.25)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Flip predictions
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        y_pred[y_pred == 0] = -1

        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"F1 Score: {f1_score(y_test, y_pred)}")

        joblib.dump(clf, 'resources/models/tfidf_isolation.joblib')

    def detect_secret(self, data):
        feed_array = [clean(article) for article in data]
        feed_string = ' '.join(feed_array)

        clf = joblib.load('resources/models/tfidf_isolation.joblib')
        vectorizer = joblib.load('resources/models/tfidf_vectorizer.joblib')

        features = vectorizer.transform([feed_string])
        prediction = clf.predict(features)

        return prediction