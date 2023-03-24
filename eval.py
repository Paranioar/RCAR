import os
import logging
import torch
import torch.nn as nn

from lib.datasets import image_caption
from lib.scanpp import SCANpp
from lib import evaluation
from lib.vocab import Vocabulary, deserialize_vocab

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(model_path, split, gpuid='0', fold5=False):
    print("use GPU:", gpuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    # load vocabulary used by the model
    if 'coco' in opt.data_name:
        vocab_file = 'coco_precomp_vocab.json'
    else:
        vocab_file = 'f30k_precomp_vocab.json'
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, vocab_file))
    vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)

    # construct model
    model = SCANpp(opt)
    model.cuda()
    model = nn.DataParallel(model)

    # load model state
    model.load_state_dict(checkpoint['model'])
    data_loader = image_caption.get_test_loader(split, opt.data_name, vocab,
                                                opt.batch_size, opt.workers, opt)

    logger.info(opt)
    logger.info('Computing results with checkpoint_{}'.format(checkpoint['epoch']))

    evaluation.evalrank(model.module, data_loader, opt, split, fold5)


if __name__ == '__main__':

    main('runs/f30k_t2i_rcar2/model_best.pth', 'test', '0', False)
    main('runs/f30k_i2t_rcar2/model_best.pth', 'test', '0', False)

    main('runs/coco_t2i_rcar2/model_best.pth', 'testall', '0', True)
    main('runs/coco_i2t_rcar2/model_best.pth', 'testall', '0', True)

    main('runs/coco_t2i_rcar2/model_best.pth', 'testall', '0', False)
    main('runs/coco_i2t_rcar2/model_best.pth', 'testall', '0', False)

    logger.info('finished')