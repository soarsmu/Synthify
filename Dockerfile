FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt -y upgrade
RUN apt -y install git vim software-properties-common tmux htop swig xvfb ffmpeg

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update && apt -y upgrade
RUN apt -y install python3.9 python3-pip python3.9-distutils python3.9-dev libssl-dev python-opengl

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

WORKDIR /root

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
# RUN pip3 install -r requirements.txt
