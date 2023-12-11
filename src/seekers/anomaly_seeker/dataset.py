import pandas as pd
import random
import string
import json

def create_random_secret(seed):
    random.seed(seed)
    return ''.join(random.choice(string.ascii_letters) for i in range(220))

def main():
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

        # Save as json
        json.dump(feed, open(save_path, 'w'), indent=4)


if __name__ == '__main__':
    main()