from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--load_checkpoint', type=str, help='path to the checkpoint file in .pth.tar format; '
                            'please do not forget to use use_class parameter when a FuseNet model with classification head is loaded')
        parser.add_argument('--results_dir', type=str, default='./results', help='test or visualization results are saved here')
        parser.add_argument('--vis_results', type=bool, default=False, help='visualize the predictions and save the comparison images')
        self.isTrain = False
        return parser
