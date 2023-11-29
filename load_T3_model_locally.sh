#!/bin/sh
wget https://huggingface.co/t5-large/resolve/main/config.json -P resources/t5-large
wget https://huggingface.co/t5-large/resolve/main/pytorch_model.bin -P resources/t5-large
wget https://huggingface.co/t5-large/resolve/main/spiece.model -P resources/t5-large
wget https://huggingface.co/t5-large/resolve/main/tokenizer.json -P resources/t5-large