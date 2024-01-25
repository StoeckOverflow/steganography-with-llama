import pandas as pd
import json
import os
from tqdm import tqdm
import glob
from src.hiders.synonym_hider import SynonymHider
from src.models import DynamicPOE

def create_dataset_with_stego_feeds(clean_feeds_path, output_path, newsfeed_or_article_labeling='newsfeed'):
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
    
    feeds = glob.glob(os.path.join(clean_feeds_path,'*.json'))
    
    num_feeds = len(feeds)
    num_arithmetic = int(num_feeds * 0.3)
    num_synonym = int(num_feeds * 0.3)
    
    doctored_articles_path = output_path
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
        
            result_newsfeed = {'feed': [], 'labels': [], 'secret':[]}
            
            if synonym_count < num_arithmetic:
                synonym_hider = SynonymHider(disable_tqdm=True)
                doctored_newsfeeds, rest_length = synonym_hider.hide_secret(feed_array, feed_secret*2, output='labeled_For_Training')
                labels = [-1] * (30 - rest_length) + [1] * rest_length
                result_newsfeed['feed'] = doctored_newsfeeds
                result_newsfeed['labels'] = labels
                result_newsfeed['secret'] = feed_secret
                synonym_count += 1
            elif arithmetic_count < num_synonym:
                try:
                    dynamic_poe = DynamicPOE(disable_tqdm=False)
                    doctored_newsfeeds, rest_length = dynamic_poe.hide(feed_secret*15, feed_array, labeled_for_training_flag=True)
                except IndexError:
                    print('Index Error')
                    labels = [1] * 30
                    result_newsfeed['feed'] = feed_array
                    result_newsfeed['labels'] = labels
                    continue
                labels = [-1] * (30 - rest_length) + [1] * rest_length
                result_newsfeed['feed'] = doctored_newsfeeds['feed']
                result_newsfeed['labels'] = labels
                result_newsfeed['secret'] = feed_secret
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

def process_chunk(chunk, file_count):
    cleaned_content = [content.strip() for content in chunk]
    grouped_content = [cleaned_content[i:i + 30] for i in range(0, len(cleaned_content), 30)]

    for group in grouped_content:
        json_object = {"feed": group}
        with open(os.path.join('resources','feeds','testfeeds_kaggle',f'feed_{file_count}.json'), 'w') as json_file:
            json.dump(json_object, json_file, indent=4)
        file_count += 1

    return file_count

def generate_clean_newsfeeds(csv_file_path):
    file_count = 0
    chunk_size = 10000

    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size, usecols=['content']):
        file_count = process_chunk(chunk['content'].tolist(), file_count)
