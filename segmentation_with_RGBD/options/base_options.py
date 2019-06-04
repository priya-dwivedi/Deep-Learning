import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--num_workers', type=int, default=1, help='number or workers for datasets loaders')
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu device id; this project currently supports only one-gpu training')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='experiment', help='checkpoints of the current experiment are saved here')

        parser.add_argument('--dataroot', required=True, help='path to dataset in h5 format; please include the type of the dataset '
                            '(nyu or sun) in the name, e.g. "nyu_db.h5 or SUN_dset.h5')
        parser.add_argument('--use_class', type=bool, default=False,
                            help='incorporate scene classification alongside semantic segmentation if True')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if self.isTrain:
            if not os.path.exists(expr_dir):
                os.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        # set gpu ids
        torch.cuda.set_device(opt.gpu_id)
        self.opt = opt
        return self.opt