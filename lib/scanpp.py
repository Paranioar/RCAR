"""VSE model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init

from lib.encoders import get_image_encoder, get_text_encoder, get_sim_encoder

import logging
logger = logging.getLogger(__name__)


class SCANpp(nn.Module):
    """
        The standard VSE model
    """

    def __init__(self, opt):
        super(SCANpp, self).__init__()
        # Build Models
        self.img_enc = get_image_encoder(opt.img_dim, opt.embed_size,
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt.vocab_size, opt.embed_size,
                                        opt.word_dim, opt.num_layers,
                                        use_bi_gru=opt.use_bi_gru,
                                        no_txtnorm=opt.no_txtnorm)
        self.sim_enc = get_sim_encoder(opt, opt.embed_size, opt.sim_dim)

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims

    def forward(self, images, captions, lengths):
        """One training step given images and captions.
        """
        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        return sims.permute(1, 0)

