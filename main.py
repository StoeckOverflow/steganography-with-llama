import json
import sys
from src.seekers.semsta_seeker.semStaSeeker import SemStaSeeker
from src.seekers.anomaly_seeker.dataset import create_newsfeed_dataset, evaluate_perplexity_threshold, plot_and_save_perplexity_statistics

if __name__ == "__main__":
     path = sys.stdin.read()
     json_data = json.loads(path)
     newsfeed = json_data['feed']
     seeker = SemStaSeeker(disable_tqdm=False)
     decision = seeker.detect_secret(newsfeed)
     print(decision)