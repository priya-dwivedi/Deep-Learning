import os
import datetime
from time import time
import numpy as np
import torch
import torch.optim
from torch.autograd import Variable
from models.fusenet_model import FuseNet
from fusenet_visualize import Visualize
from utils.utils import calculate_confusion_matrix, get_scores
import csv

class Solver(object):
    default_sgd_args = {"lr": 1e-3,
                        "momentum": 0.9,
                        "weight_decay": 0.0005}
    # default_adam_args = {"lr": 1e-4,
    #                      "betas": (0.9, 0.999),
    #                      "eps": 1e-8,
    #                      "weight_decay": 0.0}

    def __init__(self, opt, dset_info, loss_func=torch.nn.CrossEntropyLoss):
        self.opt = opt
        self.dset_name, self.seg_class_num = next(iter(dset_info.items()))
        self.use_class = opt.use_class

        self.gpu_device = opt.gpu_id
        print('[INFO] Chosen GPU Device: %s' % torch.cuda.current_device())

        # Create the FuseNet model
        self.model = FuseNet(self.seg_class_num, self.gpu_device, self.use_class)

        if opt.isTrain:
            # Set the optimizer
            optim_args = {"lr": opt.lr, "weight_decay": opt.weight_decay}
            optim_args_merged = self.default_sgd_args.copy()
            optim_args_merged.update(optim_args)
            self.optim_args = optim_args_merged
            if opt.optim.lower() == 'sgd':
                self.optim = torch.optim.SGD

            self.loss_func = loss_func()

        self.states = dict()
        self.reset_histories_and_losses()

    def reset_histories_and_losses(self):
        """
        Resets train and val histories for accuracy and the loss.
        """
        self.states['epoch'] = 0
        self.states['train_loss_hist'] = []
        self.states['train_seg_acc_hist'] = []

        if self.use_class:
            self.states['train_seg_loss_hist'] = []
            self.states['train_class_loss_hist'] = []
            self.states['train_class_acc_hist'] = []
            self.states['val_class_acc_hist'] = []

        self.states['val_seg_iou_hist'] = []
        self.states['val_seg_acc_hist'] = []
        self.states['best_val_seg_acc'] = 0.0

    def save_checkpoint(self, state, lam, is_best):
        """ Write docstring
        """
        print('[PROGRESS] Saving the model', end="", flush=True)
        checkpoint_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.dset_name.lower())
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        lam_text = ''
        if self.use_class:
            lam_text = ('_class_' + '%.5f' % lam).replace('.', '_')
        now = datetime.datetime.now()

        # If the model also the best performing model in the training session save it separately
        if is_best:
            best_model_filename = os.path.join(checkpoint_dir, 'best_model' + lam_text + '.pth.tar')
            self.states['best_model_name'] = best_model_filename
            torch.save(state, best_model_filename)
            # shutil.copyfile(checkpoint_filename, best_model_filename)
            print('\r[INFO] Best model has been successfully updated: %s' % best_model_filename)
            # shutil.copyfile(best_model_filename, checkpoint_filename)
            # print('[INFO] Checkpoint has been saved: %s' % checkpoint_filename)
            return

        # Save checkpoint with the name including epoch, - if exists, lambda value for classification - and date
        checkpoint_filename = os.path.join(checkpoint_dir, 'model_checkpoint' + lam_text + '_{}'.format(state['epoch'] + 1)
                                           + now.strftime('_%d%m%Y') + '.pth.tar')

        torch.save(state, checkpoint_filename)
        print('\r[INFO] Checkpoint has been saved: %s' % checkpoint_filename)

    def load_checkpoint(self, checkpoint_path, optim=None, only_model=False):
        """ Write docstring
        """
        if os.path.isfile(checkpoint_path):
            print('[PROGRESS] Loading checkpoint: {}'.format(checkpoint_path), end="", flush=True)

            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load the model state dictionary from the checkpoint
            self.model.load_state_dict(checkpoint['state_dict'])
            print('\r[INFO] Checkpoint has been loaded: {}'.format(checkpoint_path))

            if not only_model:
                # Load optimization method parameters from the checkpoint
                optim.load_state_dict(checkpoint['optimizer'])
                # Load the necessary checkpoint key values to the states dictionary which contains loss and history values/lists
                self.states.update({key: value for key, value in checkpoint.items() if key not in ['optimizer', 'state_dict']})

                print('[INFO] History lists have been loaded')
                print('[INFO] Resuming from epoch {}'.format(checkpoint['epoch']+1))

                # content of checkpoint is loaded to the instance; so, delete checkpoint variable to create space on the GPU
                del checkpoint
                torch.cuda.empty_cache()

                return optim
        else:
            raise FileNotFoundError('Checkpoint file not found: %s' % checkpoint_path)

    def update_learning_rate(self, optim, epoch):
        """
        Sets the learning rate to the initial LR decayed by 0.9 every 25 epochs.
        """
        lr = self.optim_args['lr'] * (0.9 ** (epoch // 25))
        for param_group in optim.param_groups:
            param_group['lr'] = lr

    def update_model_state(self, optim):
        """
        :return: dictionary of model parameters to be saved
        """
        return_dict = self.states
        return_dict.update({'state_dict': self.model.state_dict(), 'optimizer': optim.state_dict()})
        return return_dict

    def validate_model(self, val_loader, vis_results=False, outTrain=False):
        print('\n[INFO] Validating the model')
        if outTrain:
            if self.opt.isTrain:
                self.load_checkpoint(self.states['best_model_name'], only_model=True)
                print('TRAIN MODE')
            else:
                self.load_checkpoint(self.opt.load_checkpoint, only_model=True)

        # Evaluate model in eval mode
        self.model.eval()
        val_class_scores = []

        # Calculate IoU and Mean accuracy for semantic segmentation
        num_classes = self.seg_class_num
        conf_mat = np.zeros((self.seg_class_num, self.seg_class_num), dtype=np.float)
        for i, batch in enumerate(val_loader):
            val_rgb_inputs = Variable(batch[0].cuda(self.gpu_device))
            val_d_inputs = Variable(batch[1].cuda(self.gpu_device))
            val_labels = Variable(batch[2].cuda(self.gpu_device))

            print('[PROGRESS] Processing images: %i of %i    ' % (i+1, len(val_loader)), end='\r')

            if self.use_class:
                val_class_labels = Variable(batch[3].cuda(self.gpu_device))
                # Infer segmentation and classification results
                val_seg_outputs, val_class_outputs = self.model(val_rgb_inputs, val_d_inputs)
                _, val_preds_class = torch.max(val_class_outputs, 1)
                val_preds_class += 1
                val_class_scores.append(np.mean(val_preds_class.data.cpu().numpy() == val_class_labels.data.cpu().numpy()))
            else:
                val_seg_outputs = self.model(val_rgb_inputs, val_d_inputs)

            _, val_preds = torch.max(val_seg_outputs, 1)
            val_labels = val_labels - 1

            val_labels = val_labels.data.cpu().numpy()
            val_preds = val_preds.data.cpu().numpy()
            val_labels_gen_mask = val_labels >= 0

            conf_mat += calculate_confusion_matrix(val_preds, val_labels, self.seg_class_num, val_labels_gen_mask)
        global_acc, mean_acc, iou, class_mean_acc, class_iou = get_scores(conf_mat)

        if not outTrain:
            self.states['val_seg_acc_hist'].append(global_acc)
            self.states['val_seg_iou_hist'].append(iou)

            print_text = "[INFO] VALIDATION Seg_Glob_Acc: %.3f IoU: %.3f Mean_Acc: %.3f" % (global_acc, iou, mean_acc)
            print("Class Acc: ", class_mean_acc)
            print("Class IOU: ", class_iou)

            if self.use_class:
                self.states['val_class_acc_hist'].append(np.mean(val_class_scores))
                print_text += ' Class_Acc: %.3f' % self.states['val_class_acc_hist'][-1]

            print('\r[INFO] Validation has been completed       ')
            print(print_text)
            return

        print("Class Acc: ", class_mean_acc)
        np.savetxt("class_acc.csv", class_mean_acc, delimiter=",")
        print("Class IOU: ", class_iou)
        np.savetxt("class_iou.csv", class_iou, delimiter=",")
        myFile = open('stats.csv', 'w')
        writer = csv.writer(myFile)
        writer.writerow({'Global Accuracy: ', global_acc})
        writer.writerow({'IOU: ', iou})
        writer.writerow({'Mean Accuracy: ', mean_acc})
        print('[INFO] Best VALIDATION (NYU-v2) Segmentation Global Accuracy: %.3f IoU: %.3f Mean Accuracy: %.3f'
              % (global_acc, iou, mean_acc))
        print('[INFO] Orgnal. FuseNet (NYU-v2) Segmentation Global Accuracy: 0.660 IoU: 0.327 Mean Accuracy: 0.434')

        if vis_results:
            vis = Visualize(self.opt, self.model, val_loader)
            vis.visualize_predictions()

    def train_model(self, train_loader, val_loader, num_epochs=10, log_nth=0, lam=None):
        """
        Train a given model with the provided data.

        Parameters
        ----------
        train_loader:
            train data in torch.utils.data.DataLoader
        val_loader:
            validation data in torch.utils.data.DataLoader
        num_epochs: int - default: 10
            total number of training epochs
        log_nth: int - default: 0
            log training accuracy and loss every nth iteration
        lam: torch.float32
            lambda value used as weighting coefficient for classification loss
        """
        # Initiate/reset history lists and running-loss parameters
        self.reset_histories_and_losses()

        # Based on dataset sizes determine how many iterations per epoch will be done
        iter_per_epoch = len(train_loader)

        # Initiate optimization method and loss function
        optim = self.optim(self.model.parameters(), **self.optim_args)

        criterion = self.loss_func
        # Load pre-trained model parameters if resume option is chosen
        if self.opt.resume_train:
            print('[INFO] Selected training mode: RESUME')
            optim = self.load_checkpoint(self.opt.load_checkpoint, optim)
            print('[INFO] TRAINING CONTINUES')
        else:
            print('[INFO] Selected training mode: NEW')
            print('[INFO] TRAINING STARTS')

        # Determine at which epoch training session must end
        start_epoch = self.states['epoch']
        end_epoch = start_epoch + num_epochs
    
        # Start Training
        for epoch in range(start_epoch, end_epoch):
            # timestep1 = time()

            running_loss = []
            running_class_loss = []
            running_seg_loss = []
            train_seg_scores = []
            train_class_scores = []

            self.update_learning_rate(optim, epoch)

            # Train model in training mode
            self.model.train()
            for i, data in enumerate(train_loader):
                time_stamp_2 = time()

                # Zero parameter gradients
                optim.zero_grad()

                # Retrieve batch-size of input images and labels from training dataset loader
                rgb_inputs = Variable(data[0].cuda(self.gpu_device))
                d_inputs = Variable(data[1].cuda(self.gpu_device))
                train_seg_labels = Variable(data[2].cuda(self.gpu_device))

                # print("RGB and D stats: ", torch.min(rgb_inputs), torch.max(rgb_inputs), torch.min(d_inputs), torch.max(d_inputs))

                if self.use_class:
                    class_labels = Variable(data[3].cuda(self.gpu_device))
                    # forward + backward + optimize with segmentation and class loss
                    output_seg, output_class = self.model(rgb_inputs, d_inputs)
                    loss, seg_loss, class_loss = criterion(output_seg, train_seg_labels, output_class, class_labels, lambda_2=lam)
                else:
                    # forward + backward + optimize only with segmentation loss
                    output_seg = self.model(rgb_inputs, d_inputs)
                    # loss = seg_loss = criterion(output_seg, train_seg_labels)
                    loss = criterion(output_seg, train_seg_labels)

                loss.backward()
                optim.step()

                # Update running losses
                running_loss.append(loss.item())

                if self.use_class:
                    running_seg_loss.append(seg_loss)
                    running_class_loss.append(class_loss)

                    _, train_class_preds = torch.max(output_class, 1)

                    train_class_preds += 1
                    train_class_scores.append(np.mean((train_class_preds == class_labels).data.cpu().numpy()))
                    del class_labels, train_class_preds

                _, train_seg_preds = torch.max(output_seg, 1)

                labels_mask = train_seg_labels > 0
                train_seg_labels = train_seg_labels - 1

                train_seg_scores.append(np.mean((train_seg_preds == train_seg_labels)[labels_mask].data.cpu().numpy()))
                del train_seg_preds, train_seg_labels, labels_mask

                # Print statistics
                # Print each log_nth mini-batches or at the end of the epoch
                if (i+1) % log_nth == 0 or (i+1) == iter_per_epoch:
                    time_stamp_3 = time()
                    loss_log_nth = np.mean(running_loss[-log_nth:])
                    if self.use_class:
                        seg_loss_log_nth = np.mean(running_seg_loss[-log_nth:])
                        class_loss_log_nth = np.mean(running_class_loss[-log_nth:])
                        print("\r[Epoch: %d/%d Iter: %d/%d] Total_Loss: %.3f Seg_Loss: %.3f "
                              "Class_Loss: %.3f Best_Acc(IoU): %.3f LR: %.2e Lam: %.5f Time: %.2f seconds         "
                              % (epoch + 1, end_epoch, i + 1, iter_per_epoch, loss_log_nth, seg_loss_log_nth,
                                 class_loss_log_nth, self.states['best_val_seg_acc'], optim.param_groups[0]['lr'], lam,
                                 (time_stamp_3-time_stamp_2)), end='\r')
                    else:
                        print("\r[Epoch: %d/%d Iter: %d/%d] Seg_Loss: %.3f Best_Acc(IoU): %.3f LR: %.2e Time: %.2f seconds       "
                              % (epoch + 1, end_epoch, i + 1, iter_per_epoch, loss_log_nth, self.states['best_val_seg_acc'],
                                 optim.param_groups[0]['lr'], (time_stamp_3-time_stamp_2)), end='\r')

            # Log and save accuracy and loss values
            # Average accumulated loss values over the whole dataset
            self.states['train_loss_hist'].append(np.mean(running_loss))

            if self.use_class:
                self.states['train_seg_loss_hist'].append(np.mean(running_seg_loss))
                self.states['train_class_loss_hist'].append(np.mean(running_class_loss))
                # print('Train Class Scores shape and itself: ', len(train_class_scores), train_class_scores)
            # print('Train Seg Scores shape and itself: ', len(train_seg_scores), train_seg_scores)

            train_seg_acc = np.mean(train_seg_scores)
            self.states['train_seg_acc_hist'].append(train_seg_acc)

            # Run the model on the validation set and gather segmentation and classification accuracy
            self.validate_model(val_loader)

            if self.use_class:
                train_class_acc = np.mean(train_class_scores)
                self.states['train_class_acc_hist'].append(train_class_acc)

                print('[INFO] TRAIN Seg_Acc: %.3f Class_Acc: %.3f Loss: %.3f Seg_Loss: %.3f Class_Loss: %.3f Epoch: %d/%d'
                      % (train_seg_acc, train_class_acc, self.states['train_loss_hist'][-1],
                         self.states['train_seg_loss_hist'][-1], self.states['train_class_loss_hist'][-1],
                         epoch + 1, end_epoch))
            else:
                print('[INFO] TRAIN Seg_Acc: %.3f Seg_Loss: %.3f Epoch: %d/%d'
                      % (train_seg_acc, self.states['train_loss_hist'][-1], epoch + 1, end_epoch))

            # Save the checkpoint and update the model
            if (epoch+1) > self.opt.save_epoch_freq:
                current_val_seg_acc = self.states['val_seg_iou_hist'][-1]
                best_val_seg_acc = self.states['best_val_seg_acc']
                is_best = current_val_seg_acc > best_val_seg_acc
                self.states['epoch'] = epoch

                if is_best or (epoch+1) % 10 == 0:
                    self.states['best_val_seg_acc'] = max(current_val_seg_acc, best_val_seg_acc)

                    # model_state = self.update_model_state(epoch, self.model)
                    self.save_checkpoint(self.update_model_state(optim), lam, is_best)

        print('[FINAL] TRAINING COMPLETED')
        self.validate_model(val_loader, True)
