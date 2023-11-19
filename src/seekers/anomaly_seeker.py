import zipfile
import os
import json
import numpy as np
from llama_cpp import Llama
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
from transformers import AutoTokenizer

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

def get_average_token_probability(llm, tokens):
    token_probabilities = []
    for token in tokens:
        # Skip the first 5 tokens 
        if tokens.index(token) < 5:
            continue
        # Take previous tokens as prompt to get probability of next token
        prompt = ' '.join(tokens[:tokens.index(token)])
        nr_tokens_to_generate = 1000
        next_token_logprobs = llm(prompt, top_p=0, max_tokens=1, logprobs=nr_tokens_to_generate, temperature= 0, top_k=1)['choices'][0]['logprobs']['top_logprobs'][0]
        # Get probability of next token
        try:
            token_prob = np.exp(float(next_token_logprobs[token]))
        except KeyError:
            token_prob = 0.0
        print(f"Prompt: {prompt}, Token: {token}, Probability: {token_prob}")
        # Add probability to list
        token_probabilities.append(token_prob)
        print()
    return np.mean(token_probabilities)

def grammar_check(tokens):
    pos_tags = nltk.pos_tag(tokens)

    # Example how to use POS tags
    def noun_verb_frequency(pos_tags):
        pos_counts = Counter(tag for word, tag in pos_tags)
        noun_freq = pos_counts['NN'] / length
        verb_freq = pos_counts['VB'] / length
        return noun_freq, verb_freq

    # TODO: Implement grammar checks
    def check_subject_verb_agreement(pos_tags):
        # Implement specific checks here
        pass
    # Function to check tense consistency
    def check_tense_consistency(pos_tags):
        # Implement specific checks here
        pass
    def check_plural_singular_agreement(pos_tags):
        # Implement specific checks here
        pass
    def check_pronoun_antecedent_agreement(pos_tags):
        # Implement specific checks here
        pass
    def check_pronoun_case(pos_tags):
        # Implement specific checks here
        pass

    sub_verb_agreement = check_subject_verb_agreement(pos_tags)
    tense_consistency = check_tense_consistency(pos_tags)
    plural_singular_agreement = check_plural_singular_agreement(pos_tags)
    pronoun_antecedent_agreement = check_pronoun_antecedent_agreement(pos_tags)
    pronoun_case = check_pronoun_case(pos_tags)

    return sub_verb_agreement, tense_consistency, plural_singular_agreement, pronoun_antecedent_agreement, pronoun_case

def extract_features(articles):
    llm = Llama(model_path='resources/llama-2-7b.Q5_K_M.gguf', logits_all=True, verbose=False)
    features = []
    for article in articles:
        article = str(article)
        #tokens = nltk.word_tokenize(article) # TODO: use LLaMA tokenizer instead
        tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
        tokens = tokenizer.tokenize(article)
        print(tokens)

        # Article length
        length = len(article)

        # Average sentence length
        #avg_sentence_length = np.mean([len(sentence) for sentence in nltk.sent_tokenize(article)])

        # Vocabulary richness
        #vocab_richness = len(set(tokens)) / length if length > 0 else 0

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

        # Grammar Check 
        sub_verb_agreement, tense_consistency, plural_singular_agreement, pronoun_antecedent_agreement, pronoun_case = grammar_check(tokens)

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

        # Lower case count after sentence end
        #lower_case_count = sum([1 for i in range(len(tokens)) if tokens[i-1] == '.' and tokens[i].islower()])

        # Average next token probability
        #avg_token_prob = get_average_token_probability(llm, tokens)  # --> currently too slow
        avg_token_prob = 0.0

        # Collecting all features
        article_features = [length, num_special_chars, max_consecutive_special_chars, avg_token_prob]
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
    print(f"Counts: {counts}")
    if counts[1]/(counts[1]+counts[-1]) > 0.85:  # Set threshold for number of anomalies in newsfeed
        return 1
    else:
        return -1

def main():
    training_data_path = 'resources/feeds/clean_feeds.zip'
    train_model(training_data_path)

    #benign_example = 'resources/feeds/example_feed.-1'
    #result = predict_single_feed(benign_example)
    #print(f"Prediction for benign example: {result}")

    #malicious_example = 'resources/feeds/example_feed.1'
    #result = predict_single_feed(malicious_example)
    #print(f"Prediction for malicious example: {result}")

if __name__ == '__main__':
    main()
