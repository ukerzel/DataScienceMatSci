# Docker

One of the main (more technical) challenges in machine learning is that the libraries we use evolve rapidly.
This often leads us to the situation, where we require a specific setup to make sure that everything works together. Since the various libraries evolve at different paces and are developed and maintained by a diverse and unrelated group, we cannot generally assume that all versions of all packages work together.

This can be resolved by creating a controlled virtual environment using, for example, pip, poetry, or conda.
However, the machine learning libraries such as PyTorch also rely on specific versions of the underlying drivers for the graphics card (GPU) that are used to accellerate the training process.

While we can control this setup on our own system, it becomes more difficult to do so once we move to shared ressources such as a cluster - or need to run different versions.

Containers allow to go one step further than virtual environments and we can control which operating system (e.g. Ubuntu/Linux) in which version with specific libraries, etc we use.
[Docker](https://www.docker.com/) is a popular container software which has the benefit that it is well supported not only on Linux, Windows, or MacOS machines but can also be used as a starting point for running on large clusters such as the HPC cluster at the ITC at RWTH Aachen University.
The cluster systems use a different kind of container mechanism called [Apptainer](https://apptainer.org/) (formerly known as Singularity). However, one of the recommended ways to build a container image for HPC use is to start from a docker container.

The docker container is created using a "Dockerfile". In this example, we start from the official PyTorch image and add furhter machine learning library.

# Using Docker

## Building images
Before we can use the docker container, we need to build the image:
The general syntax is:
docker build -t TAG DIR
where TAG is a tag by which we identify the docker image, and DIR is the directory that contains a file called "Dockerfile" that is used to build the image.

Example:
docker build -t pytorchdatascience:v1.0 PyTorch

## Runing docker images
To run the container/image, we use "docker run"
We can either run in interactive mode or, if the container ends with a "CMD" command, this command is executed. Which version we use depends on the use-case at hand.

To start an interactive session, we use the parameters "-it".
Hence, the simplest way is to call "docker run -it TAG" where TAG is the tag we specified when building the container/image. This will give us an interactive shell inside the container - we can compare this (a bit) to logging into a remote machine.
We can also pass a script or program that should be executed as a furhter parameter.

### Accesing host files
By default, the container is fully isolated from the host system, i.e. we also do not have access to the local filesystem of the host (the machine we call "docker run" from). In many cases, this is a good thing as we do not want to risk accessing files from the image on the host.
However, in many scenarios we may want to access files on the host system, such as, for example, datafiles, program files, etc.

We can do this using, for example, so called "bind mounts" that make a directory on the host system available in the container. The general syntax is:
--mount type=bind,source=SRC_DIR,target=TARGET_DIR
where SRC_DIR is the directory on the host system that we want to make available, and TARGET_DIR is the directory at which we want to access this directory.

In the PyTorch example, we have created a local (non-root) user ("aiguru") inside the container image and then want to make the current working directory available in a directory called "bindmount":
--mount type=bind,source="${PWD}",target=/home/aiguru/bindmount

However, we need to note that the user "aiguru" is local to the container and, in general, not known to the host system. In order to avoid issues with the files we create inside the container, we need to map the (Linux) user and group ID to the values we have on the host system. We do this with the following parameter: 
--user=\`id -u\`:\`id -g\`
However, since the users on the host and docker image typically don't have the same name, we may encounter the situation that the username for this ID does not exist in the docker container. Unless we require this for our application, it is largely a cosmetic problem for interactive use.

The full command to run the docker image with a bind-mount is then:

docker run -it --name PyTorchDS --user=\`id -u\`:\`id -g\` --mount type=bind,source="${PWD}",target=/home/aiguru/bindmount pytorchdatascience:v1.0 python bindmount/PyTorch_MNIST.py

assuming that we are on the host in the local directory that contains the file PyTorch_MNIST.py that we want to execute. If we do not specify "python bindmount/PyTorch_MNIST.py", we would enter an interactive shell.


## Useful Docker commands

### Images
docker images : list images on the system
docker image rm: remove one or more images

### Build
docker build -t TAG DIR
docker tag IMAGE_ID TAG
docker push USERNAME/REPO : push to a docker repository, such as Docker Hub

### Processes
docker ps -a  : list all processes
docker ps -a -f status=exited : list all exited containers
docker rm $(docker ps -a -f status=exited -q) : remove all exited containers

### Cleanup
 docker system prune -a              : everything

### Run
docker run -it --name NAME --user=\`id -u\`:\`id -g\` --mount type=bind,source=SRC_DIR,target=TARGET_DIR TAG SCRIPT

remove "-it" for non-interactive mode

### Misc
docker exec -it <name> /bin/bash : attach an interactive shell to a running container




