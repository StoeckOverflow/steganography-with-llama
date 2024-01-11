import pandas as pd
import random
import string
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from llama_cpp import Llama
from ...utils.llama_utils import get_perplexity, get_probabilities
from src.hiders.synonym_hider import SynonymHider
from src.models import DynamicPOE
import tqdm

def evaluate_perplexity_threshold():
    llm = Llama(
            model_path='llama-2-7b.Q5_K_M.gguf',
            verbose=False,        
            logits_all=True,      
            n_ctx=512,            # Maximum context size (number of tokens) the model can handle
            n_batch=512,          # Number of tokens to process in one batch
            n_threads=3,          # Number of threads llama operations can be processed
            n_threads_batch=3,    # similar to n_threads, but for batch processing (parallel execution of different llama operations)
            use_mlock=True,        # Use mlock to prevent paging the model to disk (depends on your system's memory)
        )
    feeds_path_glob = 'resources/feeds/clean_feeds/*.json'
    feeds = glob.glob(feeds_path_glob)
    perplexity_scores = []
    i = 0
    for path in feeds:
        
        print(f"Current File: {path.split('/')[-1]}\nNumber: {i}")
        with open(path, 'r') as file:
            parsed_feed = json.load(file)
        feed_array = parsed_feed['feed']
        for feed in feed_array:
            tokenized_feed = llm.tokenizer().encode(feed)
            softmax_logits = get_probabilities(llm, tokenized_feed)
            perplexity_scores.append(get_perplexity(softmax_logits, tokenized_feed))
        i += 1
    return perplexity_scores

def plot_and_save_perplexity_statistics(perplexity_scores):
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

def create_newsfeed_dataset(newsfeed_or_article_labeling='article'):
    '''
    Create a newsfeed learning dataset for the seekers with following proportions:
    - 30% Arithmetic Encoding Hider
    - 30% Synonym Hider
    - 40 % Clean Feeds
    
    2 Options:
    newsfeed_or_article_labeling='article':
        - label each article with either 1 or -1
        - saved per newsfeed in Format. {'feed': [], 'labels': []}
    newsfeed_or_article_labeling='newsfeed'
        - label each newsfeed with either 1 or -1
        - saved per newsfeed in Format {'feed': []}
        - Label is saved in the filename doctored_feed_{formatted_number}.json;{label}
    '''
    
    feeds_path_glob = 'resources/feeds/clean_feeds/*.json'
    feeds = glob.glob(feeds_path_glob)
    
    num_feeds = len(feeds)
    num_arithmetic = int(num_feeds * 0.3)
    num_synonym = int(num_feeds * 0.3)
    
    doctored_articles_path = 'resources/feeds/doctored_feeds'
    if not os.path.exists(doctored_articles_path):
        os.makedirs(doctored_articles_path)
    
    i, arithmetic_count, synonym_count = 0, 0, 0
    if newsfeed_or_article_labeling == 'article':
        for path in tqdm.tqdm(feeds, desc="Process feeds"):
            print(f"Current File: {path}\nNumber: {i}")
            with open(path, 'r') as file:
                parsed_feed = json.load(file)
            feed_array = parsed_feed['feed']
            feed_secret = parsed_feed['secret']
            
            result_newsfeed = {'feed': [], 'labels': []}
            
            if synonym_count < num_arithmetic:
                synonym_hider = SynonymHider(disable_tqdm=True)
                doctored_newsfeeds, rest_length = synonym_hider.hide_secret(feed_array, feed_secret, output='labeled_For_Training')
                labels = [-1] * (30 - rest_length) + [1] * rest_length
                result_newsfeed['feed'] = doctored_newsfeeds
                result_newsfeed['labels'] = labels
                synonym_count += 1
            elif arithmetic_count < num_synonym:
                try:
                    dynamic_poe = DynamicPOE(disable_tqdm=True)
                    doctored_newsfeeds, rest_length = dynamic_poe.hide(feed_secret, feed_array, labeled_for_training_flag=True)
                except IndexError:
                    labels = [1] * 30
                    result_newsfeed['feed'] = feed_array
                    result_newsfeed['labels'] = labels
                    continue
                labels = [-1] * (30 - rest_length) + [1] * rest_length
                result_newsfeed['feed'] = doctored_newsfeeds['feed']
                result_newsfeed['labels'] = labels
                arithmetic_count += 1
            else: #newsfeed is not processed
                labels = [1] * 30
                result_newsfeed['feed'] = feed_array
                result_newsfeed['labels'] = labels

            formatted_number = "{:03d}".format(i)
            
            with open(os.path.join(doctored_articles_path,f"doctored_feed_{formatted_number}.json"),'w') as file:
                json.dump(result_newsfeed, file, indent=4)

            i += 1
    
    elif newsfeed_or_article_labeling == 'newsfeed':
        for path in tqdm.tqdm(feeds, desc="Process feeds"):
            print(f"Current File: {path}\nNumber: {i}")
            
            
            with open(path, 'r') as file:
                parsed_feed = json.load(file)

            feed_array = parsed_feed['feed']
            feed_secret = parsed_feed['secret']
            result_newsfeed = {'feed': []}
            
            if synonym_count < num_arithmetic:
                doctored_newsfeeds, rest_length = synonym_hider.hide_secret(feed_array, feed_secret, output='labeled_For_Training')
                label = -1
                result_newsfeed['feed'] = doctored_newsfeeds
                synonym_count += 1
            elif arithmetic_count < num_synonym:
                doctored_newsfeeds, rest_length = dynamic_poe.hide(feed_secret, feed_array, labeled_for_training_flag=True)
                label = -1
                result_newsfeed['feed'] = doctored_newsfeeds['feed']
                arithmetic_count += 1
            else: #newsfeed is not processed
                label = 1
                result_newsfeed['feed'] = feed_array

            formatted_number = "{:03d}".format(i)
            
            with open(os.path.join(doctored_articles_path, f"doctored_feed_{formatted_number}.json;{label}"), 'w') as file:
                json.dump(result_newsfeed, file, indent=4)

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