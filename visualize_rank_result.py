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

import numpy as np
from collections import OrderedDict


def extract_rank_index(sims, top_capk=5, top_imgk=5, better_mode=False,
                       include_capnum=2, save_samplenum=10):
    """
    sims: (N, 5N) matrix of similarity im-cap
    """
    assert sims.shape == (1000, 5000)
    rank_i2t_dict = {}
    rank_t2i_dict = {}

    for index in range(sims.shape[0]):
        gt_in_topK_ids = []
        gt_in_topK_pos = []

        inds = np.argsort(sims[index])[::-1]
        for k in range(5 * index, 5 * index + 5):
            if k in inds[:top_capk].tolist():
                gt_in_topK_ids.append(k)
                gt_in_topK_pos.append(inds.tolist().index(k))

        if better_mode and len(gt_in_topK_ids) >= include_capnum:
            rank_i2t_dict[index] = inds[:top_capk].tolist()
        elif not better_mode and len(gt_in_topK_ids) <= include_capnum:
            rank_i2t_dict[index] = inds[:top_capk].tolist()

        if len(rank_i2t_dict.keys()) == save_samplenum:
            break

    for index in range(sims.shape[1]):
        k = index // 5
        simsT = sims.T
        inds = np.argsort(simsT[index])[::-1]

        if better_mode and (inds.tolist().index(k) <= top_imgk):
            rank_t2i_dict[index] = inds[:top_imgk].tolist()
        elif not better_mode and (inds.tolist().index(k) >= top_imgk):
            rank_t2i_dict[index] = inds[:top_imgk].tolist()

        if len(rank_t2i_dict.keys()) == save_samplenum:
            break

    return rank_i2t_dict, rank_t2i_dict


def extract_target_index(sims_target, top_capk, top_imgk, 
                         include_capnum, anchor_i2t_dict, anchor_t2i_dict):
 
    rank_i2t_dict = OrderedDict()
    rank_t2i_dict = OrderedDict()

    for index_image in anchor_i2t_dict.keys():

        tmp_list = []
        tmp_list.append(anchor_i2t_dict[index_image])
        if_store = True
        for sims in sims_target:
            gt_in_topK_ids = []
            inds = np.argsort(sims[index_image])[::-1]
            for k in range(5 * index_image, 5 * index_image + 5):
                if k in inds[:top_capk].tolist():
                    gt_in_topK_ids.append(k)

            if len(gt_in_topK_ids) <= include_capnum:
                if_store = False
                break
            tmp_list.append(inds[:top_capk].tolist())
        
        if if_store:
            rank_i2t_dict[index_image] = tmp_list

    for index_caption in anchor_t2i_dict.keys():

        tmp_list = []
        tmp_list.append(anchor_t2i_dict[index_caption])
        if_store = True
        
        for sims in sims_target:
            simsT = sims.T
            inds = np.argsort(simsT[index_caption])[::-1]
            position = inds.tolist().index(index_caption//5)

            if index_caption//5 in anchor_t2i_dict[index_caption]:
                if position >= anchor_t2i_dict[index_caption].index(index_caption//5):
                    if_store = False
                    break
                else:
                    tmp_list.append(inds[:top_imgk].tolist())
            else:
                if position >= top_imgk:
                    if_store = False
                    break
                else:
                    tmp_list.append(inds[:top_imgk].tolist())

        if if_store:
            rank_t2i_dict[index_caption] = tmp_list

    return rank_i2t_dict, rank_t2i_dict


if __name__ == '__main__':

    # If you find this code is useful, please cite our paper and star the project. (We do need it! HaHaHaHa.)
    # Thanks for the interest in this project.

    sims_baseline = np.load('runs/baseline/results_f30k.npy', allow_pickle=True).tolist()['sims']
    sims_proposed_method1 = np.load('runs/proposed_method1/results_f30k.npy', allow_pickle=True).tolist()['sims']
    sims_proposed_method2 = np.load('runs/proposed_method2/results_f30k.npy', allow_pickle=True).tolist()['sims']

    # -------------- search the samples that are better than baseline -------------------- #
    sims_target = [sims_proposed_method1, sims_proposed_method2]
    
    anchor_i2t_dict, anchor_t2i_dict = extract_rank_index(sims_baseline, top_capk=5, top_imgk=5, better_mode=False,
                                                          include_capnum=2, save_samplenum=100)
    
    rank_i2t_dict, rank_t2i_dict = extract_target_index(sims_target, top_capk=5, top_imgk=5, include_capnum=2, 
                                                        anchor_i2t_dict=anchor_i2t_dict, anchor_t2i_dict=anchor_t2i_dict)

    print('rank_i2t_dict', rank_i2t_dict)
    print('rank_t2i_dict', rank_t2i_dict)

    # --------------------------- search the related image ------------------------------ #
    # Firstly, I resave all the test images and change their file_name in order (from xxxx.jpg to 1.jpg).
    # One can achieve this by slightly change the dataloader from https://github.com/fartashf/vsepp/blob/master/data.py
    # Then, with the above index set, We can search the correspongding images.

    # --------------------------- search the related sentence --------------------------- #
    ids = [1614, 993, 990, 3382, 994]
    captions = []
    for line in open('data/f30k/precomp/test_caps.txt', 'r'):
        captions.append(line.strip())

    for i, id in enumerate(ids):
        print('{}. '.format(i+1) + captions[id])
