"""
# Pytorch implementation for TIP2023 paper from
# https://arxiv.org/abs/2303.13371.
# "Plug-and-Play Regulators for Image-Text Matching"
# Haiwen Diao, Ying Zhang, Wei Liu, Xiang Ruan, Huchuan Lu
#
# Writen by Haiwen Diao, 2023

# If you find this code is useful, please cite our paper and star the project. (We do need it! HaHaHaHa.)
# Thanks for the interest in this project.
"""

import argparse
import os
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict, Counter


def numpy_counter_dict(sims, dicts, decimal=1, trans_to_prob=False):
    factor = np.power(10, decimal)
    sims = np.rint(sims * factor).astype(int).reshape(-1)
    counters = Counter(sims)
    dicts.update(counters)

    if trans_to_prob:
        tmp_dict = {}
        for key, value in dicts.items():
            tmp_dict[key] = value / len(sims)
        dicts.update(tmp_dict)

    return dicts


def extract_similarity(sims, capk=5, imgk=1):
    """
    sims: (N, 5N) matrix of similarity im-cap
    sim_top_capk: save the (N, capk) similarities of top-capk captions
    sim_top_imgk: save the (5N, imgk) similarities of top-imgk images
    sim_positive: save the similarities of positive image-caption pairs
    """
    assert sims.shape == (1000, 5000)
    sim_top_capk = np.zeros((sims.shape[0], capk))
    sim_non_capk = np.zeros((sims.shape[0], sims.shape[1] - capk))

    sim_top_imgk = np.zeros((sims.shape[1], imgk))
    sim_non_imgk = np.zeros((sims.shape[1], sims.shape[0] - imgk))

    sim_positive = np.zeros((sims.shape[0], 5))
    sim_negative = np.zeros((sims.shape[0], sims.shape[1] - 5))

    for index in range(sims.shape[0]):
        inds = np.argsort(sims[index])[::-1]
        sim_top_capk[index] = sims[index][inds[:capk]].copy()
        sim_non_capk[index] = sims[index][inds[capk:]].copy()

        sim_positive[index] = sims[index][5 * index:5 * index + 5].copy()
        sim_negative[index] = np.concatenate((sims[index][:5 * index], sims[index][5 * index + 5:]), axis=0).copy()

    for index in range(sims.shape[1]):
        simsT = sims.T
        inds = np.argsort(simsT[index])[::-1]
        sim_top_imgk[index] = simsT[index][inds[:imgk]].copy()
        sim_non_imgk[index] = simsT[index][inds[imgk:]].copy()

    return sim_top_capk, sim_non_capk, sim_top_imgk, sim_non_imgk, sim_positive, sim_negative


def plot_line_dict(x_keys, y_values, labels, save_path, save_name,
                   xlabels='xlabel', ylabels='ylabel', titles='title'):

    colors = ['blue', 'dodgerblue', 'Green', 'Lime', 'orange', 'Gold', 'red', 'Coral']
    # colors = ['blue', 'green', 'yellow', 'red', 'pink', 'purple']
    markers = ['s', 'o', '*', '^', 'v', '+', 'p', 'x']

    config = {
    "font.family":'serif', # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 50, # medium/large/small
    'font.style':'normal', # normal/italic/oblique
    'font.weight':'normal', # bold
    "mathtext.fontset":'cm',# 'cm' (Computer Modern)
    "font.serif": ['cmb10'], # 'Simsun'宋体
    "axes.unicode_minus": False,# 用来正常显示负号
    }
    plt.rcParams.update(config)

    plt.figure(figsize=(32, 16))
    for k, label in enumerate(labels):
        plt.plot(x_keys, y_values[k], marker=markers[k % 8], color=colors[k % 6],  label=label, linewidth=5.0)

    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)

    plt.xlabel(xlabels, fontsize=50)
    plt.ylabel(ylabels, fontsize=50)
    plt.title(titles, fontsize=50)
    plt.legend(fontsize=50)

    plt.show()
    plt.savefig(save_path + '{}.png'.format(save_name))
    plt.close()


def compute_multiple_sims(sims_list, labels, decimal=2, trans_to_prob=False, save_path=None, save_name=None, 
                          xlabels=None, ylabels=None, titles=None):
    max_value = -1e9
    min_value = 1e9
    for sims_i in sims_list:
        max_value = max(max_value, sims_i.max())
        min_value = min(min_value, sims_i.min())

    factor = np.power(10, decimal)
    x_value = np.arange(np.rint(min_value * factor).astype(int) - 1,
                        np.rint(max_value * factor).astype(int) + 1)

    y_values = []
    for sim_i in sims_list:
        temp_dict = dict.fromkeys(x_value.tolist(), 0)
        sims_dict = numpy_counter_dict(sim_i, dicts=temp_dict, decimal=decimal, trans_to_prob=trans_to_prob)
        sims_dict = OrderedDict(sorted(sims_dict.items(), key=lambda t: t[0]))
        y_values.append(sims_dict.values())

    plot_line_dict((x_value / factor).tolist(), y_values, labels, 
                    save_path, save_name, xlabels, ylabels, titles)


if __name__ == '__main__':

    # If you find this code is useful, please cite our paper and star the project. (We do need it! HaHaHaHa.)
    # Thanks for the interest in this project.

    save_name='name-of-saved-name' # need to change #
    paths = [
        'runs/model1/results_f30k.npy',
        ]

    labels = [
        'model1_label1', 
        'model1_label2',         
        ]

    sims = []
    for path in paths:
        sims.append(np.load(path, allow_pickle=True).tolist()['sims'])

    visualize_sims = []
    for sim in sims:
        visualize_sims.append(extract_similarity(sim)[index1]) # index1 related to label1
        visualize_sims.append(extract_similarity(sim)[index2]) # index2 related to label2


    compute_multiple_sims(visualize_sims, labels,
                          decimal=2,
                          trans_to_prob=True,
                          xlabels='similarity value',
                          ylabels='distribution',
                          titles='Similarity distribution of positive and negative pairs',
                          save_name=save_name, 
                          save_path='similarity_distribution/')