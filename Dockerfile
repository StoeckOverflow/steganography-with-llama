# Dockerfile for stemo:llm
FROM debian:bookworm

# Update and install dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget python3 python3-pip python3-venv time curl apt-utils git build-essential gcc

# Install Python3.11 using Homebrew and add to Path
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"
RUN brew install python@3.11

# Set the working directory and copy files into container
WORKDIR /root
COPY main.py .
COPY requirements.txt .
COPY src/ src/
COPY resources/ resources/

# Download LLM model
RUN wget -P /root/resources https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf

# Create virtual environment for image and install dependencies
ENV VIRTUAL_ENV=/root/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"
RUN ./venv/bin/pip install -r requirements.txt

CMD [ "python3.11", "main.py", "--encode" ]