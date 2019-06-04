import os
import sys
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
from torch.autograd import Variable
from models.fusenet_model import FuseNet
from utils.data_utils import get_data
from options.test_options import TestOptions


class Visualize:
    def __init__(self, opt, model=None, test_loader=None):
        self.opt = opt
        self.model = model
        self.test_loader = test_loader

        # Read the dataset and create dataset loader
        self.dset_name = os.path.basename(self.opt.dataroot)
        if self.dset_name.lower().find('nyu') is not -1:
            self.dset_info = {'NYU': 40}
        elif self.dset_name.lower().find('sun') is not -1:
            self.dset_info = {'SUN': 37}
        else:
            raise NameError('Name of the dataset file should accordingly contain either nyu or sun in it')

        # Create the scene classification ID:NAME dictionary
        self.scene_class_dict = {1: 'bedroom', 2: 'kitchen', 3: 'living room', 4: 'bathroom',
                                 5: 'dining room', 6: 'office', 7: 'home office', 8: 'classroom',
                                 9: 'bookstore', 10: 'others'}

        # Read the palette values that will be used for coloring the semantic segmentation labels, from the .txt file
        with open('./utils/text/visualization_palette.txt', 'r') as f:
            lines = f.read().splitlines()
            palette = []
        for line in lines:
            colors = line.split(', ')
            for color in colors:
                palette.append(float(color))
        self.palette = np.uint8(np.multiply(255, palette))

        self.new_image = Image.new('RGB', (960, 240))

        self.save_path = os.path.join(self.opt.results_dir, self.opt.name, 'visualization')

        # Take GPU device ID
        self.gpu_device = self.opt.gpu_id

        # By the name of the model checkpoint, check out if it contains the classification head and set use_class parameter accordingly
        self.model_path = self.opt.load_checkpoint
        self.model_name = os.path.basename(self.model_path)
        if self.model_name.lower().find('class') is not -1:
            self.opt.use_class = True

    def paint_and_save(self, image, rgb_image, scene_label, scene_pred, idx):
            """Function takes a comparision image of semantic segmentation labels, an RGB image, ground-truth and
            predicted scene classification labels, and image index. Produces a comparison image and saves it to the
            corresponding location.
            """
            x_offset = 0

            image = Image.fromarray(image, mode="P")
            image.convert("P")
            image.putpalette(self.palette)

            rgb_image = Image.fromarray(rgb_image)
            self.new_image.paste(rgb_image, (x_offset, 0))
            x_offset += rgb_image.size[0]
            self.new_image.paste(image, (x_offset, 0))

            if scene_label is not None:
                # Add scene-class names on ground truth and prediction images
                draw = ImageDraw.Draw(self.new_image)
                font = ImageFont.load_default().font
                draw.text((330, 10), ('scene class: ' + self.scene_class_dict[scene_label]), (255, 255, 255), font=font)
                draw.text((650, 10), ('scene class: ' + self.scene_class_dict[scene_pred+1]), (255, 255, 255), font=font)

            self.new_image.save(os.path.join(self.save_path, 'prediction_' + str(idx+1) + '.png'))
            print('[PROGRESS] Saving images: %i of %i     ' % (idx+1, len(self.test_loader)), end='\r')

    def visualize_predictions(self):
        """
        :return:
        """
        print('[INFO] Visualization of the results starts')
        if os.path.exists(self.save_path):
            key = input('[INFO] Taget directory already exists. You might lose previously saved images. Continue:Abort (y:n): ')
            if not key.lower() == 'y':
                print('[ABORT] Script stopped running. Images have not been saved')
                sys.exit()
        else:
            os.makedirs(self.save_path)

        if self.test_loader is None:
            _, test_data = get_data(self.opt, use_train=False, use_test=True)
            print("[INFO] %s dataset has been retrieved" % self.dset_name)

            self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
            print("[INFO] Test loader for %s dataset has been created" % self.dset_name)

        _, seg_class_num = next(iter(self.dset_info.items()))

        if self.model is None:
            # Read the FuseNet model path that will be used for prediction and load the weights to the initialized model
            self.model = FuseNet(seg_class_num, self.opt.gpu_id, self.opt.use_class)

            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("[INFO] Weights from pretrained FuseNet model has been loaded. Checkpoint: %s" % self.model_path)

        self.model.eval()

        test_class_labels = None
        test_class_preds = None

        print("[INFO] Prediction starts. Resulting comparision images will be saved under: %s" % self.save_path)
        for num, batch in enumerate(self.test_loader):
            test_rgb_inputs = Variable(batch[0].cuda(self.gpu_device))
            test_depth_inputs = Variable(batch[1].cuda(self.gpu_device))
            test_seg_labels = Variable(batch[2].cuda(self.gpu_device))

            if self.opt.use_class:
                test_class_labels = Variable(batch[3].cuda(self.gpu_device))
                # Predict the pixel-wise classification and scene classification results
                test_seg_outputs, test_class_outputs = self.model(test_rgb_inputs, test_depth_inputs)

                # Take the maximum values from the feature maps produced by the output layers for classification
                # Move the tensors to CPU as numpy arrays
                _, test_class_preds = torch.max(test_class_outputs, 1)
                test_class_labels = test_class_labels.data.cpu().numpy()[0]
                test_class_preds = test_class_preds.data.cpu().numpy()[0]
            else:
                test_seg_outputs = self.model(test_rgb_inputs, test_depth_inputs)

            # Take the maximum values from the feature maps produced by the output layers for segmentation
            # Move the tensors to CPU as numpy arrays
            _, test_seg_preds = torch.max(test_seg_outputs, 1)
            test_seg_preds = test_seg_preds.data.cpu().numpy()[0]
            test_seg_labels = test_seg_labels.data.cpu().numpy()[0]

            # Horizontally stack the predicted and ground-truth semantic segmentation labels
            comparison_images = np.hstack((np.uint8(test_seg_labels), np.uint8(test_seg_preds + 1)))

            # Move the RGB image from GPU to CPU as numpy array and arrange dimensions appropriately
            test_rgb_inputs = test_rgb_inputs.data.cpu().numpy()[0].transpose(1, 2, 0)[:, :, ::-1]

            # Color semantic segmentation labels, print scene classification labels, and save comparison images
            self.paint_and_save(comparison_images, np.uint8(test_rgb_inputs), test_class_labels, test_class_preds, num)

        print('[INFO] All %i images have been saved' % len(self.test_loader))
        print('[COMPLETED] Boring prediction images are now nice and colorful!')


if __name__ == '__main__':
    opt = TestOptions().parse()
    vis = Visualize(opt)
    vis.visualize_predictions()
