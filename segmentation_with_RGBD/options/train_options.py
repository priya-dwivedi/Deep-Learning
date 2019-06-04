from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size for the training set')
        parser.add_argument('--print_freq', type=int, default=5, help='log training accuracy and loss every nth iteration')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--resume_train', type=bool, default=False, help='continue training: load the provided model')
        parser.add_argument('--load_checkpoint', type=str, help='path to the checkpoint file in .pth.tar format')
        parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for one training session to run')
        parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate for optimizer')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for SGD')
        parser.add_argument('--lr_decay_epochs', type=int, default=25, help='multiply by a gamma every lr_decay_epoch epochs')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='momentum factor for optimizer')
        parser.add_argument('--lambda_seg', type=float, default=1.0, help='lambda value for segmentation loss')
        parser.add_argument('--lambda_class_range', type=float, default=[0.001, 0.001, 1], nargs="+",
                            help='range of lambda values for classification loss and steps in this range: (start_val, end_val, steps)')
        parser.add_argument('--optim', type=str, default='SGD',
                            help='optimization method to use during training; currently only SGD is available')
        self.isTrain = True
        return parser
