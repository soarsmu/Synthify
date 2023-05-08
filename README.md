# Synthify

This replication package contains the source code for applying Synthify and the baseline (PSY-TaLiRo) on the falsification of AI-enabled control systems.

# Environment configuration

To reproduce our experiments, machines with GPUs and NVIDIA CUDA toolkit are required.

We provide a Dockerfile to help build the experimental environment. Please run the following scripts to compile a docker image:
```
docker build -t YOUR_CUSTOM_TAG .
```

Be careful with the torch version that you need to use, modify the Dockerfile according to your cuda version pls.

Then, please run the docker:
```
docker run -it -v YOUR_LOCAL_REPO_PATH:/root/Synthify YOUR_CUSTOM_TAG
```

# How to run


# Misc

Due to the random nature of neural networks and our GA algorithm, users may obtain slightly different results. Please note that such results usually can be tolerated, i.e., they mostly do not conflict with the conclusions of the paper.