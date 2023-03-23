FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt -y upgrade
RUN apt-get -y update
RUN apt -y install software-properties-common git vim htop tmux wget

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt -y upgrade
RUN apt-get -y update
RUN apt -y install python3.7 python3-pip python3.7-distutils python3.7-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

WORKDIR /root

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

RUN pip3 install tensorflow==1.13.1 tflearn==0.5.0 protobuf==3.20.3 rtamt psy-taliro z3-solver