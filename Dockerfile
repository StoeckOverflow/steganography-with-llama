# For submission use: (For test purposes debian:bookworm)
FROM stemo:llm

COPY main.py .
COPY requirements.txt .
COPY src/ src/
COPY resources/ resources/
RUN mv llama-2-7b.Q5_K_M.gguf /resources

RUN pip install -r requirements.txt

CMD ["python3", "main.py"]