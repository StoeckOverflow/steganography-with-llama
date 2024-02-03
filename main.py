import json
import sys
#from src.hiders.synonym_hider import SynonymHider
from src.models import DynamicPOE
# from src.seekers.anomaly_seeker.anomaly_seeker import Anomaly_Seeker
# from src.seekers.anomaly_seeker.dataset import create_newsfeed_dataset, evaluate_perplexity_threshold, plot_and_save_perplexity_statistics

if __name__ == "__main__":
     # Dataset creation
     #create_newsfeed_dataset()
     #perplexities = evaluate_perplexity_threshold()
     #plot_and_save_perplexity_statistics(perplexities)
     
     # Seeker
    #  seeker = Anomaly_Seeker(disable_tqdm=False)
    #  seeker.train_model('RFC')
     #seeker.detection_interface()
     
     
     # Hider
     # data = sys.stdin.read()
     # json_data = json.loads(data)
     # feed_array = json_data['feed']
     # feed_secret = json_data['secret']
          
     #synonym_hider = SynonymHider(disable_tqdm=False)
     #synonym_hider.hide_secret(feed_array, feed_secret, output='stdout')
     
     dpoe = DynamicPOE(disable_tqdm=True, bits_per_token=2, skip_tokens=20, n_gpu_layers=-1)
     dpoe.hide_interface()
     # dpoe.recover_interface()
