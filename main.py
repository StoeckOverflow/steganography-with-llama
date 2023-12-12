#from src.models import DynamicPOE
#from src.seekers.detectGPTseeker import detectGPTseeker
from src.seekers.anomaly_seeker.anomaly_seeker import Anomaly_Seeker

if __name__ == "__main__":
    #dpoe = DynamicPOE(disable_tqdm=True)
    #dpoe.recover_interface()
    #seeker = detectGPTseeker(disable_tqdm=False)
    #seeker.detection_interface()
    seeker = Anomaly_Seeker(disable_tqdm=True)
    seeker.detection_interface()