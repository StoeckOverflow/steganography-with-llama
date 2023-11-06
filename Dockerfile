# Dockerfile for stemo:hider
FROM stemo:llm

# copy main.py, requirements.txt and src/
COPY main.py /root/
COPY ./requirements.txt /root/
COPY ./src/ /root/src/

RUN pip install -r requirements.txt

CMD [ "python3", "main.py"]
