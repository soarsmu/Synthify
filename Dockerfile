FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt -y upgrade
RUN apt -y install git vim software-properties-common tmux htop wget

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update && apt -y upgrade
RUN apt -y install python3.6 python3-pip python3.6-distutils python3.6-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

WORKDIR /root

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
# RUN pip3 install -r requirements.txt
