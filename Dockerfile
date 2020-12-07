# docker build -t ams_pytorch .

FROM nvcr.io/nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV cwd="/home/"
WORKDIR $cwd

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    vim \
    curl \
    wget \
    gfortran \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev \
    sudo \
    apt-transport-https \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    dbus-x11 \
    vlc \
    iputils-ping \
    python3-dev \
    python3-pip

# RUN apt-get install -y ffmpeg

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata python3-tk
RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

# INSTALL SUBLIME TEXT
# RUN apt install -y ca-certificates
# RUN curl -fsSL https://download.sublimetext.com/sublimehq-pub.gpg | apt-key add - && add-apt-repository "deb https://download.sublimetext.com/ apt/stable/"
# RUN apt update && apt install -y sublime-text

RUN rm -rf /var/cache/apt/archives/

### APT END ###

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN python3 -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# INSTALL PYCOCOTOOLS
RUN pip3 install --no-cache-dir cython==0.29.21
RUN pip3 install --no-cache-dir 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
