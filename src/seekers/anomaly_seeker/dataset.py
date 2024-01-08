import pandas as pd
import random
import string
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from llama_cpp import Llama
from ...utils.llama_utils import get_perplexity
from src.hiders.synonym_hider import SynonymHider
from src.models import DynamicPOE

def evaluate_perplexity_threshold(llm: Llama):
    articles_path = 'resources/feeds/clean_feeds'
    perplexity_scores = []
    i = 0
    for path in articles_path:
        
        print(f"Current File: {path.split('/')[-1]}\nNumber: {i}")
        parsed_feed = json.loads(path)
        feed_array = parsed_feed['feed']
        for feed in feed_array:
            perplexity_scores.append(get_perplexity(llm, feed))
        i += 1
    return perplexity_scores

def plot_perplexity_statistics(perplexity_scores):

    # Basic Statistical Analysis
    mean_score = np.mean(perplexity_scores)
    median_score = np.median(perplexity_scores)
    std_deviation = np.std(perplexity_scores)
    percentile_25 = np.percentile(perplexity_scores, 25)
    percentile_50 = np.percentile(perplexity_scores, 50)
    percentile_75 = np.percentile(perplexity_scores, 75)

    print(f"Mean Perplexity: {mean_score}")
    print(f"Median Perplexity: {median_score}")
    print(f"Standard Deviation: {std_deviation}")
    print(f"25th Percentile: {percentile_25}")
    print(f"50th Percentile (Median): {percentile_50}")
    print(f"75th Percentile: {percentile_75}")

    # Plot a histogram
    plt.hist(perplexity_scores, bins=20)
    plt.title("Perplexity Scores Distribution")
    plt.xlabel("Perplexity")
    plt.ylabel("Frequency")
    plt.savefig('perplexity_statistical_analysis.png')
    
    data = {
        "Statistic": ["Mean Perplexity", "Median Perplexity", "Standard Deviation", 
                    "25th Percentile", "50th Percentile (Median)", "75th Percentile"],
        "Value": [mean_score, median_score, std_deviation, 
                percentile_25, percentile_50, percentile_75]
    }

    df = pd.DataFrame(data)
    df.to_csv('resources/perplexity_statistics.csv', index=False)

def create_newsfeed_dataset():
    '''
    Create a newsfeed dataset:
    - 30% Arithmetic Encoding Hider
    - 30% Synonym Hider
    - 40 % Clean Feeds
    '''
    articles_path_glob = 'resources/feeds/clean_feeds/*.json'  # Adjust the pattern if needed
    articles = glob.glob(articles_path_glob)
    doctored_articles_path = 'resources/feeds/doctored_feeds_new'
    if not os.path.exists(doctored_articles_path):
        os.makedirs(doctored_articles_path)
    
    dynamic_poe = DynamicPOE(disable_tqdm=False)
    synonym_hider = SynonymHider(disable_tqdm=False)
    
    i = 0
    for path in articles:
        
        print(f"Current File: {path}\nNumber: {i}")
        with open(path, 'r') as file:
            parsed_feed = json.load(file)
        feed_array = parsed_feed['feed']
        feed_secret = parsed_feed['secret']
        
        if i <= 4:
            doctored_newsfeeds = synonym_hider.hide_secret(feed_array, feed_secret, output='json')
        if i > 4 and i < 8:
            doctored_newsfeeds = dynamic_poe.hide(feed_secret, feed_array)
        if i > 8:
            doctored_newsfeeds = feed_array #newsfeed is not processed

        formatted_number = "{:03d}".format(i)
        file_name = f"doctored_feed_{formatted_number}.json"
        print(f"Suggested Path: {os.path.join(doctored_articles_path, file_name)}")
        
        with open(os.path.join(doctored_articles_path,f"doctored_feed_{formatted_number}.json"),'w') as file:
            json.dump(doctored_newsfeeds, file, indent=4)

        i += 1

def create_random_secret(seed):
    random.seed(seed)
    return ''.join(random.choice(string.ascii_letters) for i in range(220))

def create_new_feeds_of_kaggle_dataset():
    more_articles_path = "resources/kaggle_articles.csv"
    dataset_df = pd.read_csv(more_articles_path, encoding="ISO-8859-1")
    # Remove content prefix
    articles_df = dataset_df['Article'].str.split(':', n=1, expand=True)[1].str.strip()
    # Filter for articles with 500 +/- 75 characters
    articles_df = articles_df[articles_df.str.len().between(385, 625)]
    # Shuffle articles
    articles_df = articles_df.sample(frac=1).reset_index(drop=True)
    # Iterate over every 30 articles
    for i in range(0, len(articles_df), 30):
        n = i % 29 + 11
        save_path = f"resources/feeds/feed_0{n}.json"

        articles = articles_df[i:i+30]
        if len(articles) < 30:
            break
        secret = create_random_secret(i)

        feed = {'feed': articles.to_list(), 'secret': secret}
        json.dump(feed, open(save_path, 'w'), indent=4)

if __name__ == '__main__':
    create_newsfeed_dataset()