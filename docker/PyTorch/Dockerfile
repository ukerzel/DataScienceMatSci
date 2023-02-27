##
## general setup 
## (this is done as root)
##

#
# start with the official (?) pytorch devel container
# https://hub.docker.com/r/pytorch/pytorch
# it seems to be released by the PyTorch team although there is no documentation
# alternatively, we could start from the NVidia one
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#
FROM  pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# say who maintains this image
LABEL maintainer="Ulrich Kerzel <ulrich.kerzel@rwth-aachen.de>"

ARG DEBIAN_FRONTEND=noninteractive

#
# here we could update the base system
# let's leave it for now
#
RUN apt-get update
# here we could update all packages
# && yes | apt-get upgrade

#
# install any systemwide software that is missing from the base image
#
RUN apt-get -y install curl





##
## Install python packages
## 
## The PyTorch container uses conda as package manager
## https://docs.conda.io/en/latest/ or https://github.com/conda/conda
##
RUN conda install -c conda-forge matplotlib=3.6.2 \
                                numba=0.56.4  \
                                numpy=1.23.5 \
                                pandas=1.5.2 \
                                scikit-learn=1.2.0 \ 
                                scikit-image=0.19.3 \
                                scipy=1.9.3 \
                                seaborn=0.12.2 \
                                shap=0.41.0 \ 
                                lime=0.2.0.1 \
                                networkx=3.0


##
## create a local user so we don't have to run as root
##
RUN addgroup --gid 1000 aiguru && \
    adduser --uid 1000 --ingroup aiguru --home /home/aiguru --shell /bin/bash --disabled-password --gecos "" aiguru

#
# One of the challenges in using docker is that the user and group IDs inside the 
# container do not match the ones on the host system
# If we then use, e.g., a bind mount to exchange files, they will have the wrong permissions
#
# We can pass the option --user=`id -u`:`id -g` to "docker run", that will fix the permissions
# However, if our user ID and/or group ID on the host system do not match with the setup we have inside
# the container, we will get errors like "I have no name" as a user-name.
# Depending on what we want to do, this is only a cosmetic problem. However, if we rely on an exisitig
# and matching user-name, it may become an issue.
# If we do not mount the host file systems, we won't have the issue anyway.
#
# The tool "fixuid" is helping with this issue
# https://github.com/boxboat/fixuid
#
# However, this is only meant for development, not production use, so we comment this out here.
#
#RUN USER=docker && \
#    GROUP=docker && \
#    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
#    chown root:root /usr/local/bin/fixuid && \
#    chmod 4755 /usr/local/bin/fixuid && \
#    mkdir -p /etc/fixuid && \
#    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml



# switch do the non-root user
USER aiguru:aiguru
WORKDIR /home/aiguru

#if we use the fixuid tool for development
#ENTRYPOINT ["fixuid"]

#
# create a local directory into which we can mount a local filesystem if needed via bind mount
# see https://docs.docker.com/storage/bind-mounts/
# the syntax is --mount type=bind,source=<source dir>,target=/home/aiguru/bindmount
# 
RUN cd $HOME
RUN mkdir $HOME/bindmount
