from src.models import DynamicPOE
from src.seekers.detectGPTseeker import detectGPTseeker

if __name__ == "__main__":
    #dpoe = DynamicPOE(disable_tqdm=True)
    #dpoe.recover_interface()
    seeker = detectGPTseeker(disable_tqdm=False)
    seeker.detection_interface()