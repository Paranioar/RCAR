"""Training script"""
import os
import time
import numpy as np
import torch

from lib.vocab import deserialize_vocab
from lib.datasets import image_caption
from lib.scanpp import SCANpp
from lib.loss import ContrastiveLoss
from lib.evaluation import evalrank, AverageMeter, LogCollector

import logging
import tensorboard_logger as tb_logger
from torch.nn.utils import clip_grad_norm_

import arguments


def main():
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuid
    device_count = len(str(opt.gpuid).split(","))

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Vocabulary
    if 'coco' in opt.data_name:
        vocab_file = 'coco_precomp_vocab.json'
    else:
        vocab_file = 'f30k_precomp_vocab.json'
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, vocab_file))
    vocab.add_word('<mask>')  # add the mask, for testing cloze
    logger.info('Add <mask> token into the vocab')
    opt.vocab_size = len(vocab)

    train_loader, val_loader = image_caption.get_loaders(
        opt.data_path, opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    model = SCANpp(opt)
    model.cuda()
    model = torch.nn.DataParallel(model)

    lr_schedules = [opt.lr_update, ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    criterion = ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
    model.Eiters = 0

    # optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another training
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))
            if opt.reset_start_epoch:
                start_epoch = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(optimizer, epoch, lr_schedules)

        if epoch >= opt.vse_mean_warmup_epochs:
            criterion.max_violation_on()

        # average meters to record the training statistics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()
        logger.info('trainable parameters: {}'.format(count_params(model)))

        end = time.time()
        for i, (images, captions, lengths, ids) in enumerate(train_loader):
            # switch to train mode
            model.train()

            # measure data loading time
            data_time.update(time.time() - end)
            model.logger = train_logger

            model.Eiters += 1
            model.logger.update('Eit', model.Eiters)
            model.logger.update('Lr', optimizer.param_groups[0]['lr'])
            # Update the model
            optimizer.zero_grad()

            if device_count != 1:
                images = images.repeat(device_count, 1, 1)

            sims = model(images, captions, lengths)
            loss = criterion(sims.t())
            model.logger.update('Le', loss.item(), sims.size(1))

            loss.backward()
            if opt.grad_clip > 0:
                clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print log info
            if model.Eiters % opt.log_step == 0:
                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(epoch, i, len(train_loader), batch_time=batch_time,
                                data_time=data_time, e_log=str(model.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
            model.logger.tb_log(tb_logger, step=model.Eiters)

        # evaluate on validation set
        rsum = evalrank(model.module, val_loader, opt, step=model.Eiters)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth'.format(epoch), prefix=opt.model_name + '/')


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(optimizer, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    main()
