import zipfile
import os
import json
import numpy as np
import nltk
import spacy
from spellchecker import SpellChecker
from collections import Counter
from sklearn.preprocessing import StandardScaler
from textstat import flesch_reading_ease
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from sklearn.svm import OneClassSVM
import joblib

def load_data(archive_path):
    articles = []
    with zipfile.ZipFile(archive_path, 'r') as zip_archive:
        for newsfeed_json in zip_archive.namelist():
            with zip_archive.open(newsfeed_json) as newsfeed_file:
                newsfeed = json.load(newsfeed_file)['feed']
                for article in newsfeed:
                    article = clean(article)
                    articles.append(article)
    return np.array(articles)

def clean(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("  ", " ")
    return text

def extract_features(articles):
    nlp = spacy.load("en_core_web_sm")
    spell = SpellChecker()
    features = []

    for article in articles:
        article = str(article)
        doc = nlp(article)
        tokens = nltk.word_tokenize(article)
        pos_tags = nltk.pos_tag(tokens)

        # Article length
        length = len(tokens)

        # Average sentence length
        avg_sentence_length = np.mean([len(sentence) for sentence in nltk.sent_tokenize(article)])

        # Vocabulary richness
        vocab_richness = len(set(tokens)) / length if length > 0 else 0

        # Readability score
        readability = flesch_reading_ease(article)

        # Named Entities Count
        num_entities = len(doc.ents)

        # POS Tag Frequencies
        pos_counts = Counter(tag for word, tag in pos_tags)
        noun_freq = pos_counts['NN'] / length
        verb_freq = pos_counts['VB'] / length

        # Spell Check (count of misspelled words)
        misspelled = spell.unknown(tokens)
        num_misspelled = len(misspelled)

        # Collecting all features
        article_features = [length, avg_sentence_length, vocab_richness, readability, num_entities, noun_freq, verb_freq, num_misspelled]
        features.append(article_features)

    scaler = StandardScaler()
    return scaler.fit_transform(features)

def train_model(path):
    newsfeeds = load_data(path)
    features = extract_features(newsfeeds)

    clf = OneClassSVM(gamma='auto', kernel='rbf')
    clf = clf.fit(features)
    joblib.dump(clf, 'resources/models/anomaly_detector.joblib')

def predict_single_feed(path):
    with open(path, 'r') as f:
        newsfeed = json.load(f)['feed']
    articles = [clean(article) for article in newsfeed]

    features = extract_features(articles)
    clf = joblib.load('resources/models/anomaly_detector.joblib')
    predictions = clf.predict(features)
    counts = Counter(predictions)
    return counts

def main():
    training_data_path = 'resources/clean_feeds/clean_feeds.zip'
    train_model(training_data_path)

    benign_example = 'resources/clean_feeds/example_feed.-1'
    result = predict_single_feed(benign_example)
    print(f"Prediction for benign example: {result}")

    malicious_example = 'resources/clean_feeds/example_feed.1'
    result = predict_single_feed(malicious_example)
    print(f"Prediction for malicious example: {result}")

if __name__ == '__main__':
    main()
