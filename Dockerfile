# FROM python:3
FROM ubuntu:18.04

# todo
# install: xdpyinfo, 

# avoid apt-get install failing on tzdata b/c of interactive prompt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# consider adding --no-install-recommends options to apt-get call
RUN apt-get install -y python3-dev python3-pip git-all
RUN apt-get install -y libgl1-mesa-glx libgl1-mesa-dri xvfb x11-apps mesa-utils
RUN apt-get install -y build-essential libx11-dev xorg-dev 
RUN apt-get install -y freeglut3-dev 
RUN apt-get install -y python3-setuptools
RUN apt-get install -y curl
RUN apt-get install -y llvm-dev xvfb

# libglu1-mesa-dev freeglut3-dev libglu1-mesa libglu1-mesa-dev libgl1-mesa-dev

RUN mkdir -p /var/tmp/build && \
    cd /var/tmp/build && \
    export VERSION=18.3.2 && \
    curl -O "https://mesa.freedesktop.org/archive/mesa-$VERSION.tar.gz" && \
    tar xfv mesa-$VERSION.tar.gz && \
    rm mesa-$VERSION.tar.gz && \
    cd mesa-$VERSION && \
    ./configure --enable-glx=gallium-xlib --with-gallium-drivers=swrast,swr --disable-dri --disable-gbm --disable-egl --enable-gallium-osmesa --prefix=/usr/local && \
    make && \
    make install && \
    cd .. && \
    rm -rf mesa-$VERSION

WORKDIR /opt

RUN git clone https://github.com/ccappelle/pyrosim && \
    cd pyrosim && \
    ./build.sh && \
    pip3 install -e .
    
