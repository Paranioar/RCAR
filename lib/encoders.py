"""VSE modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.modules.mlp import MLP

import logging

logger = logging.getLogger(__name__)


def l1norm(X, dim=-1, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)


def get_image_encoder(img_dim, embed_size, no_imgnorm=False):
    return EncoderImage(img_dim, embed_size, no_imgnorm=no_imgnorm)


def get_text_encoder(vocab_size, embed_size, word_dim, num_layers, use_bi_gru, no_txtnorm=False):
    return EncoderText(vocab_size, embed_size, word_dim, num_layers, use_bi_gru, no_txtnorm=no_txtnorm)


def get_sim_encoder(opt, embed_size, sim_dim):
    return EncoderSimilarity(opt, embed_size, sim_dim)


class EncoderImage(nn.Module):
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)
        # When using pre-extracted region features, add an extra MLP for embedding transformation
        features = self.mlp(images) + features

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


# Language Model with BiGRU
class EncoderText(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_gru, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.use_bi_gru = use_bi_gru

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)
        x_emb = self.dropout(x_emb)

        sorted_lengths, indices = torch.sort(lengths, descending=True)
        x_emb = x_emb[indices]
        inv_ix = indices.clone()
        inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)

        packed = pack_padded_sequence(x_emb, sorted_lengths.data.tolist(), batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)
        cap_emb = cap_emb[inv_ix]

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) // 2)] + cap_emb[:, :, int(cap_emb.size(2) // 2):]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # For multi-GPUs
        if cap_emb.size(1) < x_emb.size(1):
            pad_size = x_emb.size(1) - cap_emb.size(1)
            pad_emb = torch.Tensor(cap_emb.size(0), pad_size, cap_emb.size(2))
            if torch.cuda.is_available():
                pad_emb = pad_emb.cuda()
            cap_emb = torch.cat([cap_emb, pad_emb], 1)

        return cap_emb


class EncoderSimilarity(nn.Module):
    def __init__(self, opt, embed_dim, sim_dim):
        super(EncoderSimilarity, self).__init__()
        self.opt = opt
        self.embed_dim = embed_dim
        self.sim_dim = sim_dim
        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        if opt.self_regulator == 'only_rar':
            rar_step, rcr_step, alv_step = opt.rar_step, 0, 1
        elif opt.self_regulator == 'only_rcr':
            rar_step, rcr_step, alv_step = 0, opt.rcr_step, opt.rcr_step
        elif opt.self_regulator == 'coop_rcar':
            rar_step, rcr_step, alv_step = opt.rcar_step, opt.rcar_step-1, opt.rcar_step
        else:
            raise ValueError('Something wrong with opt.self_regulator')

        self.rar_modules = nn.ModuleList([Aggregation_regulator(sim_dim, embed_dim) for i in range(rar_step)])
        self.rcr_modules = nn.ModuleList([Correpondence_regulator(sim_dim, embed_dim) for j in range(rcr_step)])
        self.alv_modules = nn.ModuleList([Alignment_vector(sim_dim, embed_dim) for m in range(alv_step)])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, img_emb, cap_emb, cap_lens):

        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            query = cap_i_expand if self.opt.attn_type == 't2i' else img_emb
            context = img_emb if self.opt.attn_type == 't2i' else cap_i_expand

            smooth = self.opt.t2i_smooth if self.opt.attn_type == 't2i' else self.opt.i2t_smooth
            matrix = torch.ones(self.embed_dim)
            if torch.cuda.is_available():
                matrix = matrix.cuda()

            # ------- several setting for RAR and RCR ---------#
            if self.opt.self_regulator == 'only_rar':
                sim_mid = self.alv_modules[0](query, context, matrix, smooth)
                sim_hig = torch.mean(sim_mid, 1)
                for m, rar_module in enumerate(self.rar_modules):
                    sim_hig = rar_module(sim_mid, sim_hig)
                sim_i = self.sigmoid(self.sim_eval_w(sim_hig))

            elif self.opt.self_regulator == 'only_rcr':
                for m, rcr_module in enumerate(self.rcr_modules):
                    sim_mid = self.alv_modules[m](query, context, matrix, smooth)
                    matrix, smooth = rcr_module(sim_mid, matrix, smooth)
                wcontext = cross_attention(query, context, matrix, smooth)
                sim_i = cosine_sim(query, wcontext).mean(dim=1, keepdim=True)

            elif self.opt.self_regulator == 'coop_rcar':
                for m, rar_module in enumerate(self.rar_modules):
                    sim_mid = self.alv_modules[m](query, context, matrix, smooth)
                    if m == 0:
                        sim_hig = torch.mean(sim_mid, 1)
                    if m < (self.opt.rcar_step - 1):
                        matrix, smooth = self.rcr_modules[m](sim_mid, matrix, smooth)
                    sim_hig = rar_module(sim_mid, sim_hig)
                sim_i = self.sigmoid(self.sim_eval_w(sim_hig))

            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all


class Aggregation_regulator(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Aggregation_regulator, self).__init__()

        self.rar_q_w = nn.Sequential(nn.Linear(sim_dim, sim_dim),
                                     nn.Tanh(),
                                     nn.Dropout(0.4))
        self.rar_k_w = nn.Sequential(nn.Linear(sim_dim, sim_dim),
                                     nn.Tanh(),
                                     nn.Dropout(0.4))
        self.rar_v_w = nn.Sequential(nn.Linear(sim_dim, 1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, mid, hig):

        mid_k = self.rar_k_w(mid)
        hig_q = self.rar_q_w(hig)
        hig_q = hig_q.unsqueeze(1).repeat(1, mid_k.size(1), 1)

        weights = mid_k.mul(hig_q)
        weights = self.softmax(self.rar_v_w(weights).squeeze(2))

        new_hig = (weights.unsqueeze(2) * mid).sum(dim=1)
        new_hig = l2norm(new_hig, dim=-1)

        return new_hig


class Correpondence_regulator(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Correpondence_regulator, self).__init__()

        self.rcr_smooth_w = nn.Sequential(nn.Linear(sim_dim, sim_dim // 2),
                                          nn.Tanh(),
                                          nn.Linear(sim_dim // 2, 1))
        self.rcr_matrix_w = nn.Sequential(nn.Linear(sim_dim, sim_dim * 2),
                                          nn.Tanh(),
                                          nn.Linear(sim_dim * 2, embed_dim))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, matrix, smooth):

        matrix = (self.tanh(self.rcr_matrix_w(x)) + matrix).clamp(min=-1, max=1)
        smooth = self.relu(self.rcr_smooth_w(x) + smooth)

        return matrix, smooth


class Alignment_vector(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Alignment_vector, self).__init__()

        self.sim_transform_w = nn.Linear(embed_dim, sim_dim)

    def forward(self, query, context, matrix, smooth):

        wcontext = cross_attention(query, context, matrix, smooth)
        sim_rep = torch.pow(torch.sub(query, wcontext), 2)
        sim_rep = l2norm(self.sim_transform_w(sim_rep), dim=-1)

        return sim_rep


def cross_attention(query, context, matrix, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    query = torch.mul(query, matrix)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, dim=-1)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL)
    attn = F.softmax(attn*smooth, dim=2)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    wcontext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    wcontext = torch.transpose(wcontext, 1, 2)
    wcontext = l2norm(wcontext, dim=-1)

    return wcontext


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_regulator', default='only_rar',
                        help='only_rar, only_rcr, coop_rcar')
    parser.add_argument('--rar_step', default=1, type=int,
                        help='step of RCR')
    opt = parser.parse_args()

    model = EncoderSimilarity(opt, 1024, 256)
    para = count_params(model)
    para1 = count_params(model.alv_modules)
    para2 = count_params(model.rar_modules)
    print(para1 + para2)
    print(para)