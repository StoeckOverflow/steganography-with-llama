# steganography-with-llama

steganography-with-llama is a Python-based tool that utilizes a large language model (LLAMA) to perform steganography, the art of hiding information within other data, such as images or text. This project allows you to encode and decode secret messages within various media files using the power of advanced natural language processing.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

1. Clone the repository:

```shell
git clone https://github.com/yourusername/llama-steganography.git
cd llama-steganography
```

Install the required dependencies:
```shell
pip install -r requirements.txt
```

Download the LLAMA language model weights (if not included in the repository) and place them in the appropriate directory.

You're ready to start using LLAMA Steganography!

## Usage
To use the hiders (encoder and decoder) and the seekers, you firstly have to set up the main.py.

### File format
The input file always has to be in JSON format.The Input and Output format for the hiders encoding mechanism should look like this:

```shell
Input:
    {"secret": "<base64 encoded data>", "feed": ["<article 1>", "<article 2>", …, "<article 30>"]}
Output:
    {"feed": ["<article 1>", "<article 2>", …, "<article 30>"]} 
```

The Input and Output for the hiders recover mechanism should look like this:
```shell
Input:
    {"feed": ["<article 1>", "<article 2>", …, "<article 30>"]} 
Output:
     {"secret": "<base64 encoded data>"} 
```

The Input and Output for the seekers should loo like this:
```shell
Input:
    {"feed": ["<article 1>", "<article 2>", …, "<article 30>"]} 
Output:
    {"result": true | false} 
```

### Usage Hider
First you have to define, wheater you want to start the encoder or hider in the main.py. All Hiders use a specific interface, so that you can write the main as follows to use them:

```shell
from src.models import DynamicPOE

if __name__ == "__main__":
    dpoe = DynamicPOE(disable_tqdm=False)
    dpoe.hide_interface() #Use dpoe.recover_interface() for recovering a message
```

### Usage Seeker
To use the seeker, you'll have to rewrite the define the main like this:

```shell
from src.seekers.anomaly_seeker.anomaly_seeker import Anomaly_Seeker

if __name__ == "__main__":
    seeker = Anomaly_Seeker(disable_tqdm=True)
    seeker.detection_interface()
```

All seekers use the detection interface. To use a different seeker you just have to change the seeker implementation in the main.py.

After you adjusted the main.py you can run the program using this command on the shell:

```shell
cat input_media_file.json | python3 main.py
```

For more advanced options and customization, consult the documentation or run the scripts with the --help flag.

# Disclaimer
LLAMA Steganography is intended for educational and research purposes only. Use it responsibly and comply with all applicable laws and regulations when hiding or extracting information from media files.
