FROM debian:bookworm

# Update and install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y wget python3 python3-pip python3-venv curl apt-utils git build-essential gcc

# Install Homebrew and add to Path
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"

# Install Python 3.11 using Homebrew
RUN brew install python@3.11

# Set the working directory and copy files into container
WORKDIR /app

COPY main.py .
COPY requirements.txt .
COPY src/ src/
COPY resources/ resources/

# Create virtual environment for image and install dependencies
RUN python3.11 -m venv venv
ENV PATH="/app/venv/bin:${PATH}"
RUN pip install -r requirements.txt

# Download LLM model
# RUN wget -P /app/resources https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf

CMD [ "python3.11", "main.py", "--decode"]