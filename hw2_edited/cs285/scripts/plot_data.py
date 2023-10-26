#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:15:32 2022

@author: oyindamola
"""
import numpy as np
from numpy import genfromtxt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp


DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data(data, xaxis="Epoch", value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)
    # sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)


    sns.lineplot(data=data, x=xaxis, y=value, ci='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    # plt.legend(loc='best').set_draggable(True)
    plt.legend(loc='upper center', ncol=3, handlelength=1,
              borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page,
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

file = '/home/oyindamola/Research/homework_fall2021/hw2/data/default/q2_pg_q1_lb_no_rtg_dsa_CartPole-v0_17-02-2022_09-05-12/run-q2_pg_q1_sb_no_rtg_dsa_CartPole-v0_17-02-2022_09-01-10-tag-Train_AverageReturn.csv'

std_file = '/home/oyindamola/Research/homework_fall2021/hw2/data/default/q2_pg_q1_lb_no_rtg_dsa_CartPole-v0_17-02-2022_09-05-12/run-q2_pg_q1_sb_no_rtg_dsa_CartPole-v0_17-02-2022_09-01-10-tag-Train_StdReturn.csv'

data = pd.read_csv(file)

std_data = pd.read_csv(std_file)

data =data.rename(columns={"Wall Time": "Wall Time", "Step": "Epochs", "Value": "Average Episode Return"})
plot_data(data, xaxis="Epochs", value="Average Episode Return",smooth=2)


# import the necessary packages

# folder = '/home/oyindamola/Research/homework_fall2021/hw2/data/'
# import os
# if __name__ == "__main__":
#     for (root,dirs,files) in os.walk(folder):
#         print (root)
#         print (dirs)
#         print (files)
#         print ('--------------------------------')
