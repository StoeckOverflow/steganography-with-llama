import zipfile
from tqdm import tqdm
import json
import numpy as np
from llama_cpp import Llama
from collections import Counter
from itertools import groupby
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

def create_dataset(dir_path_benign, dir_path_malicious):
    benign_articles = load_data(dir_path_benign)
    benign_labels = np.ones(len(benign_articles))
    malicious_articles = load_data(dir_path_malicious)
    malicious_labels = np.ones(len(malicious_articles)) * -1

    df_benign = pd.DataFrame({'article': benign_articles, 'label': benign_labels})
    df_malicious = pd.DataFrame({'article': malicious_articles, 'label': malicious_labels})
    df = pd.concat([df_benign, df_malicious])
    return df

def shannon_entropy(article):
    prob = [ float(article.count(c)) / len(article) for c in dict.fromkeys(list(article)) ]
    entropy = - sum([ p * np.log2(p) for p in prob ])
    return entropy

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def special_chars_count(article):
    special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?']
    return sum([article.count(char) for char in special_chars])

def extract_features(articles):
    llm = Llama(model_path='resources/llama-2-7b.Q5_K_M.gguf', logits_all=True, verbose=False)
    features = []
    for article in tqdm(articles, desc='Extracting features'):
        article = str(article)
        tokens = llm.tokenizer().encode(article)

        # Article length
        length = len(article)

        # Vocabulary richness
        vocab_richness = len(set(tokens)) / length if length > 0 else 0

        # Special characters count
        num_special_chars = special_chars_count(article)

        # Shannon entropy
        entropy = shannon_entropy(article)

        # Average token probability
        llm.reset()
        llm.eval(tokens)
        logits = np.array(llm._scores)
        token_probabilities = softmax(logits)
        avg_token_probability = np.mean(token_probabilities)

        # Collecting all features
        article_features = [length, vocab_richness, num_special_chars, entropy, avg_token_probability]
        features.append(article_features)

    scaler = StandardScaler()
    transformed_features = scaler.fit_transform(features)
    df = pd.DataFrame(transformed_features, columns=['length', 'vocab_richness', 'num_special_chars', 'entropy', 'avg_token_probability'])
    return df

def train_model(benign_data_path, malicious_data_path):
    print('Training model...')
    dataset = create_dataset(benign_data_path, malicious_data_path)
    articles = dataset['article']
    labels = dataset['label']
    features = extract_features(articles)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=420, stratify=labels)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf = clf.fit(X_train, y_train)

    f1 = f1_score(y_test, clf.predict(X_test))
    print(f"F1 Score: {f1}")

    joblib.dump(clf, 'resources/models/anomaly_detector.joblib')

def predict_single_feed(path):
    print('Predicting news feed...')
    with open(path, 'r') as f:
        newsfeed = json.load(f)['feed']
    articles = [clean(article) for article in newsfeed]

    features = extract_features(articles)
    clf = joblib.load('resources/models/anomaly_detector.joblib')
    predictions = clf.predict(features)
    counts = Counter(predictions)
    print('Predictions:', predictions)
    if counts[1]/(counts[1]+counts[-1]) > 0.6:  # Set threshold for number of anomalies in newsfeed
        return 1
    else:
        return -1

def main():
    benign_data_path = 'resources/feeds/clean_feeds.zip'
    malicious_data_path = 'resources/doctored_feeds/doctored_feeds.zip'
    train_model(benign_data_path, malicious_data_path)

    benign_example = 'resources/feeds/feed_003.json'
    result = predict_single_feed(benign_example)
    print(f"Prediction for benign example: {result}")
    print('True label: 1')

    print()

    malicious_example = 'resources/doctored_feeds/doctored_feed_003.json'
    result = predict_single_feed(malicious_example)
    print(f"Prediction for malicious example: {result}")
    print('True label: -1')

if __name__ == '__main__':
    main()
