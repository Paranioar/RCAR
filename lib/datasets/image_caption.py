"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
import random
import nltk

import logging

logger = logging.getLogger(__name__)


class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, vocab, opt, train):
        self.vocab = vocab
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')

        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        # Image features
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        caption = self.captions[index]

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        target = process_caption(self.vocab, caption, self.train)
        image = torch.Tensor(self.images[img_index])
        return image, target, index, img_index

    def __len__(self):
        return self.length


def process_caption(vocab, caption, drop=False):
    if not drop:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = list()
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        return torch.Tensor(caption)
    else:
        # Convert caption (string) to word ids.
        tokens = ['<start>', ]
        tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
        tokens.append('<end>')
        deleted_idx = []
        for i, token in enumerate(tokens):
            if token in ['<start>', '<end>']:
                tokens[i] = vocab(token)
                continue
            prob = random.random()
            if prob < 0.20:
                prob /= 0.20
                # 50% randomly change token to mask token
                if prob < 0.5:
                    tokens[i] = vocab.word2idx['<mask>']
                # 10% randomly change token to random token
                elif prob < 0.6:
                    tokens[i] = random.randrange(len(vocab))
                # 40% randomly remove the token
                else:
                    tokens[i] = vocab(token)
                    deleted_idx.append(i)
            else:
                tokens[i] = vocab(token)

        if len(deleted_idx) != 0:
            tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]

        return torch.Tensor(tokens)


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images
    images = torch.stack(images, 0)

    # Merge captions
    lengths = torch.LongTensor([len(cap) for cap in captions])
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_loader(data_path, data_name, data_split, vocab, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dset = PrecompRegionDataset(data_path, data_name, data_split, vocab, opt, train)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              drop_last=train)

    return data_loader


def get_loaders(data_path, data_name, vocab, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', vocab, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, data_name, 'dev', vocab, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, vocab, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', vocab, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, vocab, opt,
                             batch_size, False, workers, train=False)
    return test_loader
