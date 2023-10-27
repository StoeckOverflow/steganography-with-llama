# Dockerfile for stemo:llm
FROM debian:bookworm

# install dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget python3 python3-pip python3-venv time

# use root dir for our setup
WORKDIR /root

# setup virtual env for python and make it easily available for later use
ENV VIRTUAL_ENV=/root/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies
RUN ./venv/bin/pip install llama_cpp_python typing tqdm

# download llm
RUN wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf
