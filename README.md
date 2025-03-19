# GA_RIMS cluster

cluster + singularity + PBS

cluster folder:

- bash file to submit a job
- bash file to submit an array of jobs

singularity folder:

- container definition (.def)
- container image (.sif)

### Usage

clone repository

```
git clone https://github.com/kenny2817/GA_RIMS.git ./folder_name
cd folder_name
```

checkout in the cluster branch

```
git checkout cluster
```

make bash files executable

```
cd cluster
make
```

add diagram folder and adjust the name in the GA_RA_PST.py
