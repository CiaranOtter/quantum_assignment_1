FROM ubuntu:latest

RUN apt-get update -y && apt-get install -y python3 python3-pip
RUN apt install python3-venv -y 
WORKDIR /deps

RUN python3 -m venv quantum-venv

WORKDIR /root
COPY ./ .

ENV PATH="/deps/quantum-venv/bin:$PATH"

RUN pip install -r requirements.txt
