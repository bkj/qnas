FROM nvidia/cuda:8.0-cudnn5-devel

# Install dependencies
RUN apt-get -y update
RUN apt-get install -y git wget

# Install anaconda
RUN wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
RUN bash Anaconda2-4.4.0-Linux-x86_64.sh -b -p /root/.anaconda
RUN rm Anaconda2-4.4.0-Linux-x86_64.sh
RUN echo 'export PATH="/root/.anaconda/bin:$PATH"' >> ~/.bashrc 
ENV PATH="/root/.anaconda/bin:${PATH}"

# Install packages
RUN pip install tqdm keras
RUN conda install -y pytorch torchvision cuda80 -c soumith
RUN conda install -y tensorflow-gpu
RUN conda install -y -c conda-forge awscli
RUN apt install -y jq

# BUG: !! Install most recent version of torchvision
RUN conda uninstall torchvision
RUN pip install git+git://github.com/pytorch/vision

# Install redis
RUN apt install -y redis-server
RUN pip install redis rq

# Download qnas + datasets
RUN mkdir -p /root/projects/
WORKDIR /root/projects
RUN git clone https://github.com/bkj/qnas
WORKDIR /root/projects/qnas/
RUN ./utils/download.sh

RUN git pull