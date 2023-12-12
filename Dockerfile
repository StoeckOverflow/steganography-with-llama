FROM stemo:llm

COPY main.py .
COPY requirements.txt .
COPY src/ src/
COPY resources/ resources/

RUN pip install -r requirements.txt

RUN python3 -m spacy download en_core_web_sm

CMD ["python3", "main.py"]