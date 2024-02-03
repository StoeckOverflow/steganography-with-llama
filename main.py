from src.seekers.anomaly_seeker.anomaly_seeker import Anomaly_Seeker

if __name__ == "__main__":
     seeker = Anomaly_Seeker(disable_tqdm=False)
     #seeker.train_model(modelName='OCSVM')
     seeker.detection_interface()