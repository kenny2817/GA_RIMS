# Singularity

Singularity is a container platform. It allows you to create and run containers that package up pieces of software in a way that is portable and reproducible

## build image

to get all the dependecy i used a docker configured as sepecified in the Dockckerfile present in this folder

to build the docker:

docker build -t singularity-builder .

to run the docker:

sudo docker run --privileged -it -v singularity/:/app -w /app singularity-builder

to build singularity image from recipe:

sudo singularity build container.sif container.def

## run container

it is possible to add arguments to both the run and the program, in this case:

singularity exec --bind folder/:/data --pwd /data singularity/container.sif python3 /data/program.py

- exec -> executes the command in the cntainer
- --bind -> creates a volume within the container from your machine
- --pwd -> sets the working directory
