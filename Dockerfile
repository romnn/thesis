FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

WORKDIR /src
RUN apt-get update && apt-get install -y \
    build-essential \
    curl

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
