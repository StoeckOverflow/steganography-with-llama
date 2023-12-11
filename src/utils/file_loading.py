import json
import numpy as np
import pandas as pd
import zipfile
from .string_modification import clean

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