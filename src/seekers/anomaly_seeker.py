import zipfile
import os
import json
import numpy as np
import nltk
import spacy
from spellchecker import SpellChecker
from collections import Counter
from itertools import groupby
from sklearn.preprocessing import StandardScaler
from textstat import flesch_reading_ease
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
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
    text = text.replace("-", " ")
    return text

def extract_features(articles):
    features = []

    for article in articles:
        article = str(article)
        tokens = nltk.word_tokenize(article)

        # Article length
        length = len(article)

        # Average sentence length
        avg_sentence_length = np.mean([len(sentence) for sentence in nltk.sent_tokenize(article)])

        # Vocabulary richness
        vocab_richness = len(set(tokens)) / length if length > 0 else 0

        # Readability score
        #readability = flesch_reading_ease(article)
        #print(f'Readability score: {readability}')  # --> found this to be irrelevant

        # Named Entities Count
        #nlp = spacy.load("en_core_web_sm")
        #doc = nlp(article)
        #num_entities = len(doc.ents)
        #print(f'Named Entities Count: {num_entities}')
        #print("\tText:", [(ent.text) for ent in doc.ents])
        #print("\tEntities:", [(ent.text, ent.label_) for ent in doc.ents])  # --> found this to be irrelevant

        # POS (Parts of Speech) Tag Frequencies
        pos_tags = nltk.pos_tag(tokens)
        pos_counts = Counter(tag for word, tag in pos_tags)
        noun_freq = pos_counts['NN'] / length
        verb_freq = pos_counts['VB'] / length
        # Maybe use POS Tags for grammar checks instead of frequency?

        # Spell Check (count of misspelled words)
        #spell = SpellChecker()
        #spell.word_frequency.load_words([]) # Add custom words here
        #misspelled = spell.unknown(tokens)
        #num_misspelled = len(misspelled)
        #print(f'Misspelled words: {num_misspelled} ({misspelled})')  # --> found this to be irrelevant (too many false positives)

        # Special characters count
        special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?']
        num_special_chars = sum([article.count(char) for char in special_chars])

        # Max consecutive special characters
        max_consecutive_special_chars = max([len(list(g)) for k, g in groupby(article) if k in special_chars])

        # Collecting all features
        article_features = [length, avg_sentence_length, vocab_richness, noun_freq, verb_freq, num_special_chars, max_consecutive_special_chars]
        features.append(article_features)

    scaler = StandardScaler()
    transformed_features = scaler.fit_transform(features)
    return transformed_features

def train_model(path):
    articles = load_data(path)
    features = extract_features(articles)

    #clf = OneClassSVM(gamma='auto', kernel='rbf')
    clf = IsolationForest(random_state=420)
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
    if counts[1]/(counts[1]+counts[-1]) > 0.85:  # Set threshold for number of anomalies in newsfeed
        return 1
    else:
        return -1

def main():
    training_data_path = 'resources/feeds/clean_feeds.zip'
    train_model(training_data_path)

    benign_example = 'resources/feeds/example_feed.-1'
    result = predict_single_feed(benign_example)
    print(f"Prediction for benign example: {result}")

    malicious_example = 'resources/feeds/example_feed.1'
    result = predict_single_feed(malicious_example)
    print(f"Prediction for malicious example: {result}")

if __name__ == '__main__':
    main()
