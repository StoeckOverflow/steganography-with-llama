import json
import numpy as np
import pandas as pd
import zipfile

def create_dataset(dir_path_benign, dir_path_malicious):
    benign_articles = load_data(dir_path_benign)
    benign_labels = np.ones(len(benign_articles))
    malicious_articles = load_data(dir_path_malicious)
    malicious_labels = np.ones(len(malicious_articles)) * -1

    df_benign = pd.DataFrame({'article': benign_articles, 'label': benign_labels})
    df_malicious = pd.DataFrame({'article': malicious_articles, 'label': malicious_labels})
    df = pd.concat([df_benign, df_malicious])
    return df

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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()