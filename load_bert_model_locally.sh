#!/bin/sh
wget https://huggingface.co/bert-base-cased/resolve/main/config.json -P resources/bert-base-cased
wget https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin -P resources/bert-base-cased
wget https://huggingface.co/bert-base-cased/resolve/main/tokenizer_config.json -P resources/bert-base-cased
wget https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json -P resources/bert-base-cased
wget https://huggingface.co/bert-base-cased/resolve/main/vocab.txt -P resources/bert-base-cased