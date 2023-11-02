FROM debian:bookworm

# Set non-interactive environment variable to prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y wget python3 python3-pip python3-venv curl apt-utils git

# Install Homebrew
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add Homebrew to the PATH
ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"

# Install Python 3.11 using Homebrew
RUN brew install python@3.11

# Set the working directory
WORKDIR /app

# Copy your Python script and requirements file into the container
COPY main.py .
COPY requirements.txt .
COPY src/ src/

# Use a virtual environment for Python dependencies
RUN python3.11 -m venv venv
ENV PATH="/app/venv/bin:${PATH}"

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Download LLM model
RUN wget -P /app/resources https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf

# Set the entry point to your Python script
CMD [ "python3.11", "main.py" ]