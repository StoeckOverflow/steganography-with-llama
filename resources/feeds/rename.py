import os

path = "/home/stud04/steganography-with-llama-1/resources/feeds/kaggle/testfeeds_kaggle_doctored"

for filename in os.listdir(path):
    if filename.endswith(".json"):
        os.rename(os.path.join(path, filename), os.path.join(path, filename.strip("doctored_")+";1"))
    elif filename.endswith(";1"):

        os.rename(os.path.join(path, filename), os.path.join(path, filename.strip("doctored_")))
    else:
        continue