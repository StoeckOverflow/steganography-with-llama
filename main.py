#from src.seekers.semsta_seeker.dataset_generator import datasetGenerator
#from src.seekers.semsta_seeker.data_augmentor import DataAugmentor
from src.seekers.semsta_seeker.semStaSeeker import SemStaSeeker
if __name__ == "__main__":
     #path = sys.stdin.read()
     #json_data = json.loads(path)
     #newsfeed = json_data['feed']
     #seeker = SemStaSeeker(disable_tqdm=False)
     #decision = seeker.detect_secret(newsfeed)
     #print(decision)
     
     #generator = datasetGenerator()
     #generator.generate_clean_newsfeeds('resources/datasets/articles2.csv')
     #generator.create_dataset_with_stego_feeds(clean_feeds_path='resources/feeds/testfeeds_kaggle',
     #                                          output_path='resources/feeds/testfeeds_kaggle_doctored')
     
     semStaSeeker = SemStaSeeker()
     semStaSeeker.detection_interface()
     #data_augmentor = DataAugmentor()
     #data_augmentor.create_augmented_datasets(directory_path='resources/feeds/doctored_feeds_newsfeeds_copy',
     #                                         save_dir='resources/feeds/doctored_feeds_newsfeeds_copy')
     #semStaSeeker.train_and_evaluate_model(newsfeeds_dir='resources/feeds/testfeeds_kaggle_doctored')
     #semStaSeeker.train_and_evaluate_model_on_saved_data()
     #semStaSeeker.train_and_evaluate_model_bootstrapped(newsfeeds_dir='resources/testfeeds/kaggle_doctored')
     #semStaSeeker.train_and_evaluate_model_bootstrapped(newsfeeds_dir='resources/feeds/doctored_feeds_newsfeeds')