import os
import datetime
import torch
from torch.utils import data
from fusenet_solver import Solver
from utils.data_utils import get_data
from utils.loss_utils import cross_entropy_2d
from options.train_options import TrainOptions
from utils.utils import print_time_info


if __name__ == '__main__':
    opt = TrainOptions().parse()

    dset_name = os.path.basename(opt.dataroot)
    if dset_name.lower().find('nyu') is not -1:
        dset_info = {'NYU': 40}
    elif dset_name.lower().find('sun') is not -1:
        dset_info = {'SUN': 37}
    else:
        raise NameError('Name of the dataset file should accordingly contain either nyu or sun in it')

    print('[INFO] %s dataset is being processed' % list(dset_info.keys())[0])
    train_data, test_data = get_data(opt, use_train=True, use_test=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    print("[INFO] Data loaders for %s dataset have been created" % list(dset_info.keys())[0])

    if opt.use_class:
        # Grid search for lambda values
        # Lambda is the coefficient of the classification loss
        # i.e.: total_loss = segmentation_loss + lambda * classification_loss
        start, end, steps = opt.lambda_class_range
        lambdas = torch.linspace(start, end, steps=int(steps)).cuda(opt.gpu_id)

        for i, lam in enumerate(lambdas):
            start_date_time = datetime.datetime.now().replace(microsecond=0)
            print('[INFO] Training session: [%i of %i]' % (i+1, steps))
            print('[INFO] Lambda value for this training session: %.5f' % lam)

            solver = Solver(opt, dset_info, loss_func=cross_entropy_2d)
            solver.train_model(train_loader, test_loader, num_epochs=opt.num_epochs, log_nth=opt.print_freq, lam=lam)

            end_date_time = datetime.datetime.now().replace(microsecond=0)
            print_time_info(start_date_time, end_date_time)
    else:
        # Run an individual training session
        start_date_time = datetime.datetime.now().replace(microsecond=0)
        solver = Solver(opt, dset_info, loss_func=cross_entropy_2d)
        solver.train_model(train_loader, test_loader, num_epochs=opt.num_epochs, log_nth=opt.print_freq)
        end_date_time = datetime.datetime.now().replace(microsecond=0)
        print_time_info(start_date_time, end_date_time)
