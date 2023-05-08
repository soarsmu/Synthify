# Synthify

This replication package contains the source code for applying *Synthify* and the baseline (*PSY-TaLiRo*) on the falsification of AI-enabled control systems. The plant of each control system is in `envs` and the AI controllers are in `checkpoints`. Moreover, the hyperparameters of each control systems are in `config.json`.

# Environment configuration

To reproduce our experiments, we provide a Dockerfile to help build the experimental environment. Please run the following scripts to compile a docker image:
```
docker build -t YOUR_CUSTOM_TAG .
```
Then, please run the docker as a container, and mount your code folder to a folder in your container.
```
docker run -it -v YOUR_LOCAL_REPO_PATH:/root/Synthify YOUR_CUSTOM_TAG
```

# How to run

**Setting 1**: Given the same number of falsification trials, i.e., 50 trials, we provide two bash scripts to run *Synthify* and *PSY-TaLiRo* on the four AI-enabled control systems, respectively. Note that the script will repeat experiments for 10 times on each control system. Please run the following scripts to reproduce the results of *Synthify*:
```
bash synthify.sh
```
It contains the following command for each control system:
```
python3 falsify.py --env=ENV
```
Please run the following scripts to reproduce the results of *PSY-TaLiRo*:
```
bash baseline.sh
```
It contains the following command for each control system:
```
python3 psy_taliro.py --env=ENV
```

We strongly recommend users to run the above bash scripts, as they help save the results of each trial in a separate file for subsequent analysis. Otherwise, results of each trial will be printed on the screen, and users need to manually save them for subsequent analysis.

**Setting 2**: Given the same time budget, i.e., 10 minutes, we provide the following command to run *Synthify* and *PSY-TaLiRo* on the four AI-enabled control systems, respectively. Note that the script will repeat experiments for 10 times on each control system.
 Please run the following scripts to reproduce the results of *Synthify*:
```
bash synthify_time.sh
```
It contains the following command for each control system:
```
python3 falsify_time_budget.py --env=ENV
```
Please run the following scripts to reproduce the results of *PSY-TaLiRo*:
```
bash baseline_time.sh
```
It contains the following command for each control system:
```
python3 psy_taliro_time_budget.py --env=ENV
```
We strongly recommend users to run the above bash scripts, as they help save the results of each trial in a separate file for subsequent analysis. Otherwise, results of each trial will be printed on the screen, and users need to manually save them for subsequent analysis.

**Parse results**: After saving the results of each trial into `results` folder, please run the following command to parse the results:
```
python3 parse_results.py
```
The parsed results will be printed on the screen with *average results* and *standard deviation*, as well as *significance test results* with *effect sizes*.

# Misc

Since we report the average results of 10 experiments in the paper, and the falsification algorithms are stochastic, users may get slightly different results. Please note that such results usually can be tolerated, i.e., they mostly do not conflict with the conclusions of the paper. We also provide the code for significance test results with effect sizes to help users analyze the results.

If users want to train the AI controllers by themselves, please run the following command:
```
python3 DDPG.py --env=ENV
```
But we do not recommend users to do so, as it may take a long time to train the AI controllers. We have provided the trained AI controllers in `checkpoints` folder.