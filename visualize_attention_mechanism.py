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

import os
import cv2 as cv
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from imageio import imread


def visualize_region(img, bboxes, size, attention, save_path, save_name, margin=-1, alpha=0.5):
    # the number of regions
    k = bboxes.shape[0]
    assert k == 36

    mask = np.zeros((size['image_h'], size['image_w']))

    for i in range(k):
        bbox = bboxes[i]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if attention[i] > margin:
            mask_atten = attention[i] * np.ones((y2-y1, x2-x1))
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], mask_atten)

    mask = np.uint8((mask[:, :, np.newaxis]).repeat(3, axis=-1) * 255)
    # mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_LINEAR)
    heat_img = cv.applyColorMap(mask, cv.COLORMAP_JET)
    merge_img = cv.addWeighted(img, alpha, heat_img, 1-alpha, 0)
    cv.imwrite(os.path.join(save_path, '{}.png'.format(save_name)), merge_img)


def visualize_word(x, y_value, x_key, save_path, save_name, figsize=[20, 10], labelsize=30):

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

    plt.figure(figsize=figsize)
    plt.bar(x, y_value, tick_label=x_key, align='center')
    plt.tick_params(axis='x', labelsize=labelsize)
    plt.tick_params(axis='y', labelsize=labelsize)
    plt.xticks(rotation=30)

    plt.show()
    plt.savefig(os.path.join(save_path,'{}.png'.format(save_name)))
    plt.close()


if __name__ == '__main__':

    # If you find this code is useful, please cite our paper and star the project. (We do need it! HaHaHaHa.)
    # Thanks for the interest in this project.
    
    # -----------------------initial path settings----------------------------- #
    root_image = 'data/f30k/flickr30k-images/'
    root_image_id = 'data/f30k/precomp/test_ids.txt'
    root_caption = 'data/f30k/precomp/test_caps.txt'

    # load from https://github.com/Paranioar/RCAR/tree/main/data/f30k/id_mapping.json
    root_image_id_mapping = 'data/f30k/id_mapping.json'

    # load from https://drive.google.com/file/d/1ZVLIN7uSh3dqYAEldelyYF2ei9vicJvZ/view
    img_bboxes = np.load('data/f30k/precomp/test_ims_bbx.npy') 
    img_sizes = np.load('data/f30k/precomp/test_ims_size.npy', allow_pickle=True)

    caption_list = []
    with open(root_caption, 'r') as f:
        for line in f:
            caption_list.append(line.strip())
    
    with open(root_image_id, 'r') as f:
        ids = f.readlines()
        image_ids = [int(x.strip()) for x in ids]

    with open(root_image_id_mapping, 'r') as f:
        id_to_path = json.load(f)


    cap_id_list, img_id_list = [], []
    for i in range(100):
        cap_id_list.append(i)
        img_id_list.append(i//5)

    model_name = 'name-of-the-saved-model'  # need to change #

    # ---------------------- initial visual settings--------------------------- #
    for cap_id, img_id in zip(cap_id_list, img_id_list):
        # load raw word including <start> and <end>
        caption = caption_list[cap_id]
        print(caption)
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        word_list = []
        word_list.append('_start_')
        word_list.extend([str(token) for token in tokens])
        word_list.append('_end_')

        # load raw image, img_bbox, img_size
        image = cv.imread(os.path.join(root_image, id_to_path[str(image_ids[img_id*5])]))
        # cv.imshow('image', image)
        # cv.waitKey(0)
        img_bbox = img_bboxes[img_id*5]
        img_size = img_sizes[img_id*5]

        # ------------------------ word to region -------------------------------- #
        file_name = '{}_sentence{}_image{}'.format(model_name, cap_id, img_id)
        load_path = 'attention/word2region/weights/'
        save_path = 'attention/word2region/images/{}'.format(file_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        attention = np.load('{}{}.npy'.format(load_path, file_name))
        print('word2region/caption{}'.format(cap_id), attention.shape)
        assert len(word_list) == attention.shape[0]
        
        # shape: (n_word, n_region)
        for word_id, word in enumerate(word_list):
            print(word)
            visualize_region(image, img_bbox, img_size, attention[word_id], save_path, 
                            '{}{}_region_sentence{}_image{}'.format(word_id, word, cap_id, img_id))

        # ----------------------- region to word ------------------------------- #
        file_name = '{}_sentence{}_image{}'.format(model_name, cap_id, img_id)
        load_path = 'attention/region2word/weights/'
        save_path = 'attention/region2word/images/{}'.format(file_name)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        attention = np.load('{}{}.npy'.format(load_path, file_name))
        print('region2word/image{}'.format(img_id), attention.shape)
        
        # the number of regions = 36
        k = img_bbox.shape[0]
        assert k == 36
        
        x = np.arange(0, len(word_list), 1)
        x_key = np.array(word_list)
        
        # generate region mask to visual which region
        for region_id in range(k):
            mask = np.ones((img_size['image_h'], img_size['image_w'])) * 0.3
            bbox = img_bbox[region_id]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            mask[y1:y2, x1:x2] = np.ones((y2 - y1, x2 - x1))
            alpha = mask.reshape((img_size['image_h'], img_size['image_w'], 1)) * 255
            new_img = np.concatenate((image, alpha), -1)
            cv.imwrite(os.path.join(save_path, 'region{}_im{}.png'.format(region_id, img_id)), new_img)
        
            # generate the attention map on sentence words
            visualize_word(x, attention[region_id], x_key, save_path, 
                           '{}region_word_sentence{}_image{}'.format(region_id, cap_id, img_id), figsize=[25, 15])








