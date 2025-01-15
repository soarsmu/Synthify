import logging
import numpy as np

logging.getLogger().setLevel(logging.INFO)
from scipy.stats import mannwhitneyu
from numpy import mean, sqrt, std


import itertools as it

from bisect import bisect_left
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as ss

from pandas import Categorical


def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000

    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/

    :param treatment: a numeric list
    :param control: another numeric list

    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data d and f must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (
        2 * n * m
    )  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


def VD_A_DF(data, val_col: str = None, group_col: str = None, sort=True):
    """

    :param data: pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.
    :param val_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains values.
    :param group_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains group names.
    :param sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.

    :return: stats : pandas DataFrame of effect sizes

    Stats summary ::
    'A' : Name of first measurement
    'B' : Name of second measurement
    'estimate' : effect sizes
    'magnitude' : magnitude

    """

    x = data.copy()
    if sort:
        x[group_col] = Categorical(
            x[group_col], categories=x[group_col].unique(), ordered=True
        )
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)

    groups = x[group_col].unique()

    # Pairwise combinations
    g1, g2 = np.array(list(it.combinations(np.arange(groups.size), 2))).T

    # Compute effect size for each combination
    ef = np.array(
        [
            VD_A(
                list(x[val_col][x[group_col] == groups[i]].values),
                list(x[val_col][x[group_col] == groups[j]].values),
            )
            for i, j in zip(g1, g2)
        ]
    )

    return pd.DataFrame(
        {
            "A": np.unique(data[group_col])[g1],
            "B": np.unique(data[group_col])[g2],
            "estimate": ef[:, 0],
            "magnitude": ef[:, 1],
        }
    )


def cohen_d(x, y):
    return (mean(x) - mean(y)) / sqrt(
        ((std(x, ddof=1) + 0.001) ** 2 + (std(y, ddof=1) + 0.001) ** 2) / 2.0
    )


iters = 10
# for env in ["cartpole", "pendulum", "quadcopter", "self_driving"]:
for env in [
    "cartpole",
    "pendulum",
    "quadcopter",
    "self_driving",
    "lane_keeping",
    "car_platoon_4",
    "car_platoon_8",
    "oscillator",
]:
    # for env in ["car_platoon_8", "cartpole", "quadcopter", "pendulum", "self_driving", "lane_keeping", "car_platoon_4", "oscillator"]:
    iter_mean = []
    iter_median = []
    iter_mean_all = []
    iter_median_all = []
    rate = []
    sim_time = []
    fal_time = []
    coverage = []
    no_sim_rate = []

    iter_mean_syn = []
    iter_mean_syn_lin = []
    iter_median_syn = []
    iter_mean_all_syn = []
    iter_mean_all_syn_lin = []
    iter_median_all_syn = []
    rate_syn = []
    sim_time_syn = []
    fal_time_syn = []
    coverage_syn = []
    no_sim_rate_syn = []

    for num in range(1, iters + 1):
        with open("psy_taliro_" + env + "_" + str(num) + ".log") as f:
            iter_mean_temp = 0
            iter_median_temp = 0
            rate_temp = 0
            lines = f.readlines()
            for line in lines:
                if "INFO:root:coverage of slice specifications is" in line:
                    coverage.append(float(line.split(" ")[-1]))
                if "NFO:root:falsification rate wrt. 50 trials is" in line:
                    # if float(line.split(" ")[-1]) < 1: print(env, num)
                    rate_temp = float(line.split(" ")[-1])
                    rate.append(rate_temp)
                if (
                    "INFO:root:mean number of simulations over successful trials is"
                    in line
                ):
                    iter_mean_temp = float(line.split(" ")[-1])
                if (
                    "INFO:root:median number of simulations over successful trials"
                    in line
                ):
                    iter_median_temp = float(line.split(" ")[-1])
                if "INFO:root:simulation time" in line:
                    sim_time.append(float(line.split(" ")[-1]))
                if "INFO:root:falsification time" in line:
                    fal_time.append(float(line.split(" ")[-1]))
                if "INFO:root:non-simulation time ratio" in line:
                    no_sim_rate.append(float(line.split(" ")[-1]))
            iter_mean.append(iter_mean_temp)
            iter_median.append(iter_mean_temp)
            iter_mean_all.append(iter_mean_temp * rate_temp + 300 * (1 - rate_temp))
        try:
            with open("synthify_" + env + "_" + str(num) + ".log") as f:
                iter_mean_temp = 0
                iter_median_temp = 0
                rate_temp = 0
                iter_mean_syn_lin_temp = 0
                syn_time = 0
                lines = f.readlines()
                for line in lines:
                    if "INFO:root:Synthesis time:" in line:
                        syn_time = float(line.split(" ")[-1])
                    if "INFO:root:coverage of slice specifications is" in line:
                        coverage_syn.append(float(line.split(" ")[-1]))
                    if "NFO:root:falsification rate wrt. 50 trials is" in line:
                        # if float(line.split(" ")[-1]) < 1: print(env, num)
                        rate_temp = float(line.split(" ")[-1])
                        rate_syn.append(rate_temp)
                    if (
                        "INFO:root:mean number of simulations over successful trials is"
                        in line
                    ):
                        iter_mean_temp = float(line.split(" ")[-1])
                    if (
                        "INFO:root:median number of simulations over successful trials"
                        in line
                    ):
                        iter_median_temp = float(line.split(" ")[-1])
                    if (
                        "INFO:root:mean number of linear simulations over successful trials"
                        in line
                    ):
                        iter_mean_syn_lin_temp = float(line.split(" ")[-1])
                    sim_time_syn_temp = 0
                    if "INFO:root:linear simulation time" in line:
                        sim_time_syn_temp += float(line.split(" ")[-1])
                    if "INFO:root:DRL simulation time" in line:
                        sim_time_syn_temp += float(line.split(" ")[-1])
                    if "INFO:root:falsification time" in line:
                        fal_time_syn.append(float(line.split(" ")[-1]) + syn_time)
                    if "INFO:root:non-simulation time ratio" in line:
                        no_sim_rate_syn.append(float(line.split(" ")[-1]))
                sim_time_syn.append(sim_time_syn_temp)
                iter_mean_syn.append(iter_mean_temp)
                iter_median_syn.append(iter_mean_temp)
                iter_mean_all_syn.append(
                    iter_mean_temp * rate_temp + 300 * (1 - rate_temp)
                )
                iter_mean_syn_lin.append(iter_mean_syn_lin_temp)
                iter_mean_all_syn_lin.append(
                    iter_mean_syn_lin_temp * rate_temp + 300 * (1 - rate_temp)
                )
        except:
            pass

        # print(rate)
        # print(rate_syn)

    logging.info("\t env: %s", env)
    # logging.info("\t\t baseline's coverage: %f, with standard deviation %f \n\t\t\t our coverage: %f, with standard deviation %f \n\t\t\t improvement: %f ", mean(coverage), np.std(coverage, ddof=1), mean(coverage_syn), np.std(coverage_syn, ddof=1), (mean(coverage_syn)-mean(coverage))/mean(coverage))
    logging.info(
        "\t\t baseline falsification rate: %f, with standard deviation %f \n\t\t\t our falsification rate: %f, with standard deviation %f \n\t\t\t improvement: %f ",
        mean(rate) * 50,
        np.std(rate, ddof=1),
        mean(rate_syn) * 50,
        np.std(rate_syn, ddof=1),
        (mean(rate_syn) - mean(rate)) / mean(rate),
    )
    # logging.info("\t\t baseline mean number of simulations over successful trials: %f, with standard deviation %f \n\t\t\t our mean number of simulations over successful trials: %f, with standard deviation %f \n\t\t\t improvement: %f", mean(iter_mean), np.std(iter_mean, ddof=1), mean(iter_mean_syn), np.std(iter_mean_syn, ddof=1), (mean(iter_mean_syn)-mean(iter_mean))/mean(iter_mean))
    # logging.info("\t\t baseline mean number of simulations over all trials: %f, with standard deviation %f, our mean number of simulations over all trials: %f, with standard deviation %f, our mean number of linear simulations over all trials: %f, with standard deviation %f, ", mean(iter_mean_all), np.std(iter_mean_all, ddof=1), mean(iter_mean_all_syn), np.std(iter_mean_all_syn, ddof=1), mean(iter_mean_all_syn_lin), np.std(iter_mean_all_syn_lin, ddof=1))
    # logging.info("\t\t baseline simulation time is: %f, with standard deviation %f \n\t\t\t our simulation time is: %f, with standard deviation %f \n\t\t\t improvement: %f", mean(sim_time), np.std(sim_time, ddof=1), mean(sim_time_syn), np.std(sim_time_syn, ddof=1), (mean(sim_time_syn)-mean(sim_time))/mean(sim_time))
    logging.info(
        "\t\t baseline falsification time is: %f, with standard deviation %f \n\t\t\t our falsification time is: %f, with standard deviation %f \n\t\t\t improvement: %f",
        mean(fal_time) / mean(rate) / 50,
        np.std(fal_time, ddof=1),
        mean(fal_time_syn) / mean(rate_syn) / 50,
        np.std(fal_time_syn, ddof=1),
        (mean(fal_time_syn) - mean(fal_time)) / mean(fal_time),
    )
    # logging.info("\t\t non-simulation time ratio is: %f, with standard deviation %f \n\t\t\t our non-simulation time ratio is: %f, with standard deviation %f \n\t\t\t improvement: %f", mean(no_sim_rate), np.std(no_sim_rate, ddof=1), mean(no_sim_rate_syn), np.std(no_sim_rate_syn, ddof=1), (mean(no_sim_rate_syn)-mean(no_sim_rate))/mean(no_sim_rate))
    # print(coverage,coverage_syn)
    # logging.info("\t\t coverage's significance: %s, A 12:: %f %s", mannwhitneyu(coverage, coverage_syn, method="asymptotic"), VD_A(coverage_syn, coverage[:len(coverage_syn)])[0], VD_A(coverage_syn, coverage[:len(coverage_syn)])[1])
    # if len(rate) == 0:
    #     rate = [0.0]*len(rate_syn)
    # else:
    #     rate = [600/x/50 for x in rate]
    # rate_syn = [600/x/50 for x in rate_syn]
    print(rate, rate_syn)
    logging.info(
        "\t\t FR's significance: %s, A 12: %f, %s",
        mannwhitneyu(rate, rate_syn, method="asymptotic"),
        VD_A(rate_syn, rate[: len(rate_syn)])[0],
        VD_A(rate_syn, rate[: len(rate_syn)])[1],
    )
    logging.info(
        "\t\t simulation time's significance: %s, A 12: %f %s",
        mannwhitneyu(fal_time, fal_time_syn, method="asymptotic"),
        VD_A(fal_time[: len(fal_time_syn)], fal_time_syn)[0],
        VD_A(fal_time[: len(fal_time_syn)], fal_time_syn)[1],
    )


# a = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# c = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

# print(mannwhitneyu(a, b, method="asymptotic"))
# print(mannwhitneyu(c, b, method="asymptotic"))
