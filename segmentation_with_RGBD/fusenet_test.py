import os
import datetime
import torch
from torch.utils import data
from fusenet_solver import Solver
from utils.data_utils import get_data
from options.test_options import TestOptions
from utils.utils import print_time_info


if __name__ == '__main__':
    opt = TestOptions().parse()

    if opt.load_checkpoint.lower().find('class') is not -1:
        opt.use_class = True
        print('USE CLASS:', opt.use_class)

    dset_name = os.path.basename(opt.dataroot)
    if dset_name.lower().find('nyu') is not -1:
        dset_info = {'NYU': 40}
    elif dset_name.lower().find('sun') is not -1:
        dset_info = {'SUN': 37}
    else:
        raise NameError('Name of the dataset file should accordingly contain either nyu or sun in it')

    print('[INFO] %s dataset is being processed.' % list(dset_info.keys())[0])
    _, test_data = get_data(opt, use_train=False, use_test=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    print("[INFO] Data loaders for %s dataset have been created" % list(dset_info.keys())[0])

    start_date_time = datetime.datetime.now().replace(microsecond=0)

    print('[INFO] Inference starts')
    solver = Solver(opt, dset_info)
    solver.validate_model(test_loader, opt.vis_results, True)

    end_date_time = datetime.datetime.now().replace(microsecond=0)
    print_time_info(start_date_time, end_date_time)
