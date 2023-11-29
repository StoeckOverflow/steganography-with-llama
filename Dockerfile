# For submission use: (For test purposes debian:bookworm)
FROM stemo:llm

COPY main.py .
COPY requirements.txt .
COPY src/ src/
COPY resources/ resources/
RUN mv llama-2-7b.Q5_K_M.gguf /resources

# Download T5 3B model for Pertubations in DetectGPT.py
# Model size: 3gb
RUN wget https://huggingface.co/t5-large/resolve/main/config.json -P resources/t5-large
RUN wget https://huggingface.co/t5-large/resolve/main/pytorch_model.bin -P resources/t5-large
RUN wget https://huggingface.co/t5-large/resolve/main/spiece.model -P resources/t5-large
RUN wget https://huggingface.co/t5-large/resolve/main/tokenizer.json -P resources/t5-large

RUN pip install -r requirements.txt

CMD ["python3", "main.py"]