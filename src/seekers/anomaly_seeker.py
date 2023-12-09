import zipfile
from tqdm import tqdm
import json
import numpy as np
from llama_cpp import Llama
from textblob import TextBlob
import spacy
from collections import Counter
from itertools import groupby
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os

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

def flesch_reading_ease(article):
    words = article.split()
    num_words = len(words)
    num_sentences = article.count('. ') + article.count('! ') + article.count('? ')
    num_syllables = 0
    for word in words:
        if word:  # Ensure the word is not empty
            syllables = count_syllables(word)
            num_syllables += syllables
    if num_words > 0 and num_sentences > 0:
        flesch_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words) # Formula for Flesch-Kincaid reading ease
        return flesch_score
    else:
        return 0

def count_syllables(word):
    if not word or not word.strip(): 
        return 0

    vowels = 'aeiouy'
    num_syllables = 0
    word = word.lower().strip(".:;?!")
    
    if len(word) > 0 and word[0] in vowels:
        num_syllables += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            num_syllables += 1
    if word.endswith('e'):
        num_syllables -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        num_syllables += 1
    if num_syllables == 0:
        num_syllables = 1
    return num_syllables

def shannon_entropy(article):
    prob = [float(article.count(c)) / len(article) for c in dict.fromkeys(list(article))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def special_chars_count(article):
    special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?']
    return sum([article.count(char) for char in special_chars])

def sentiment_consistency(article):
    sentences = article.split('. ')
    sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    variance = np.var(sentiments)
    return variance

def named_entity_analysis(article, nlp=spacy.load("en_core_web_sm")):
    doc = nlp(article)
    entities = set([ent.text for ent in doc.ents])
    return len(entities)

def repetition_patterns(article, n=3):
    tokens = article.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    freq = Counter(ngrams)
    # Count n-grams that appear more than once
    repetitions = sum(1 for item in freq.values() if item > 1)
    return repetitions

def count_transition_words(article):
    transition_words = ['however', 'furthermore', 'therefore', 'consequently', 'meanwhile', 'nonetheless', 'moreover', 'likewise', 'instead', 'nevertheless', 'otherwise', 'similarly', 'accordingly', 'subsequently', 'hence', 'thus', 'still', 'then', 'yet', 'accordingly', 'additionally', 'alternatively', 'besides', 'comparatively', 'conversely', 'finally', 'further', 'furthermore', 'hence', 'however', 'indeed', 'instead', 'likewise', 'meanwhile', 'moreover', 'nevertheless', 'next', 'nonetheless', 'otherwise', 'similarly', 'still', 'subsequently', 'then', 'therefore', 'thus', 'whereas', 'while', 'yet'] 
    count = sum(article.count(word) for word in transition_words)
    return count

def extract_features(articles):
    llm = Llama(model_path='resources/llama-2-7b.Q5_K_M.gguf', logits_all=True, verbose=False)
    features = []
    for article in tqdm(articles, desc='Extracting features'):
        article = str(article)
        tokens = llm.tokenizer().encode(article)

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
        transition_words = count_transition_words(article)

        # Average token probability
        llm.reset()
        llm.eval(tokens)
        logits = np.array(llm._scores)
        token_probabilities = softmax(logits)
        avg_token_probability = np.mean(token_probabilities)

        # Token probability variance
        token_probability_variance = np.var(token_probabilities)

        # Collecting all features
        article_features = [length, avg_sentence_length, 
                            type_token_ratio, 
                            flesch_score, 
                            vocab_richness, 
                            num_special_chars, 
                            entropy, sentiment, 
                            named_entities, 
                            repetition, 
                            transition_words,
                            avg_token_probability,
                            token_probability_variance
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
                                                     'transition_words',
                                                     'avg_token_probability',
                                                     'token_probability_variance'
                                                     ])
    return df

def train_model(benign_data_path, malicious_data_path):
    print('Training model...')
    dataset = create_dataset(benign_data_path, malicious_data_path)
    articles = dataset['article']
    labels = dataset['label']
    features = extract_features(articles)

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
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    # Calculate False Positive Rate
    FP = np.sum(np.logical_and(y_pred == 1, y_test == -1))
    TN = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    FPR = FP / (FP + TN)
    print(f"FPR: {FPR}")
    
    joblib.dump(clf, 'resources/models/anomaly_detector.joblib')

    return FPR

def predict_single_feed(path, FPR=1/30):
    print('Predicting news feed...')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            newsfeed = json.load(f)['feed']
    except UnicodeDecodeError:
        with open(path, 'r', encoding='ISO-8859-1') as f:
            newsfeed = json.load(f)['feed']
    articles = [clean(article) for article in newsfeed]

    features = extract_features(articles)
    clf = joblib.load('resources/models/anomaly_detector.joblib')
    predictions = clf.predict(features)
    counts = Counter(predictions[:6])
    print('Predictions:', predictions)

    if counts[-1] > 30 * FPR:
        return -1
    else:
        return 1

def main():
    #benign_data_path = 'resources/feeds/clean_feeds.zip'
    #malicious_data_path = 'resources/doctored_feeds/doctored_feeds.zip'
    #FPR = train_model(benign_data_path, malicious_data_path)
    FPR = 1/30

    malicious_dir_path = 'resources/doctored_feeds/'
    for feed in os.listdir(malicious_dir_path):
        if feed.endswith('.zip'):
            continue
        feed_path = os.path.join(malicious_dir_path, feed)
        result = predict_single_feed(feed_path, FPR)
        true_label = -1
        print(f"Prediction for {feed}: {result} ({result == true_label})")
        print()

    benign_dir_path = 'resources/feeds/'
    for feed in os.listdir(benign_dir_path):
        if feed.endswith('.zip'):
            continue
        feed_path = os.path.join(benign_dir_path, feed)
        result = predict_single_feed(feed_path)
        true_label = 1
        print(f"Prediction for {feed}: {result} ({result == true_label})")
        print()

if __name__ == '__main__':
    main()
