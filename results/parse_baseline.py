from scipy.stats import mannwhitneyu
from numpy import mean, sqrt, std
def cohen_d(x,y):
    return (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)
for env in ["cartpole", "pendulum", "quadcopter", "self_driving"]:
    iter_mean = 0
    iter_median = 0
    rate = 0
    sim_time = 0
    fal_time = 0
    coverage = 0
    for num in range(1, 21):
        with open("baseline_"+env+"_"+str(num)+".log") as f:
            lines = f.readlines()
            for line in lines:
                if "INFO:root:coverage of slice specifications is" in line:
                    coverage += float(line.split(" ")[-1])
                if "NFO:root:falsification rate wrt. 50 trials is" in line:
                    if float(line.split(" ")[-1]) < 1: print(env, num)
                    rate += float(line.split(" ")[-1])
                if "INFO:root:mean number of simulations over successful trials is" in line:
                    iter_mean += float(line.split(" ")[-1])
                if "INFO:root:median number of simulations over successful trials" in line:
                    iter_median += float(line.split(" ")[-1])
    print(env, coverage/20, rate/20, iter_mean/20, iter_median/20)

a = [0.75] * 20
b = [1.0] * 12 + [0.75] * 8
print(mannwhitneyu(a, b, method="asymptotic"), cohen_d(a, b))
    

