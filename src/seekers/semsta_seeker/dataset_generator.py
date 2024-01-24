import pandas as pd
import json
import os

def process_chunk(chunk, file_count):
    
    cleaned_content = [content.strip() for content in chunk]
    grouped_content = [cleaned_content[i:i + 30] for i in range(0, len(cleaned_content), 30)]

    for group in grouped_content:
        json_object = {"feed": group}
        with open(os.path.join('resources','feeds','testfeeds_kaggle',f'feed_{file_count}.json'), 'w') as json_file:
            json.dump(json_object, json_file, indent=4)
        file_count += 1

    return file_count

def main(csv_file_path):
    file_count = 0
    chunk_size = 10000

    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size, usecols=['content']):
        file_count = process_chunk(chunk['content'].tolist(), file_count)

if __name__ == "__main__":
    main('resources/datasets/articles3.csv')