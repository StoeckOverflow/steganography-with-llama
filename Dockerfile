# Base Image BEGIN
# Dockerfile for stemo:llm
FROM debian:bookworm

# Update and install dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget python3 python3-pip python3-venv time

# Install Python3.11 using Homebrew and add to Path
#RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"
#RUN brew install python@3.11

# Set the working directory and copy files into container
WORKDIR /root

# Download LLM model
RUN wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf

# Create virtual environment for image and install dependencies
ENV VIRTUAL_ENV=/root/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"

RUN pip install llama_cpp_python
# Base Image END

COPY main.py .
COPY requirements.txt .
COPY src/ src/
COPY resources/ resources/
RUN mv llama-2-7b.Q5_K_M.gguf /resources

RUN ./venv/bin/pip install -r requirements.txt

CMD ["python3", "main.py", "--encode"]