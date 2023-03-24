import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/f30k',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./data/vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--gpuid', default='0', type=str,
                        help='The number of gpuid')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--logger_name', default='./runs/test/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/test',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=0,
                        help='The number of warmup epochs using mean vse loss')
    parser.add_argument('--reset_start_epoch', action='store_true',
                        help='Whether restart the start epoch when load weights')
    parser.add_argument('--use_bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')

    parser.add_argument('--attn_type', default='t2i',
                        help='{t2i,i2t}')
    parser.add_argument('--t2i_smooth', default=10.0, type=float,
                        help='The value of t2i softmax lambda')
    parser.add_argument('--i2t_smooth', default=3.0, type=float,
                        help='The value of i2t softmax lambda')

    parser.add_argument('--self_regulator', default='coop_rcar',
                        help='only_rar, only_rcr, coop_rcar')
    parser.add_argument('--rcar_step', default=0, type=int,
                        help='step of RCR cooperation with RAR')
    parser.add_argument('--rcr_step', default=0, type=int,
                        help='step of RCR')
    parser.add_argument('--rar_step', default=0, type=int,
                        help='step of RAR')

    return parser
