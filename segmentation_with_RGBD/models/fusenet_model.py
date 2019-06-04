import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class FuseNet(nn.Module):
    def __init__(self, num_labels, gpu_device=0, use_class=True):
        super(FuseNet, self).__init__()

        # Load pre-trained VGG-16 weights to two separate variables.
        # They will be used in defining the depth and RGB encoder sequential layers.
        feats = list(models.vgg16(pretrained=True).features.children())
        feats2 = list(models.vgg16(pretrained=True).features.children())

        # Average the first layer of feats variable, the input-layer weights of VGG-16,
        # over the channel dimension, as depth encoder will be accepting one-dimensional
        # inputs instead of three.
        avg = torch.mean(feats[0].cuda(gpu_device).weight.data, dim=1)
        avg = avg.unsqueeze(1)

        bn_moment = 0.1
        self.use_class = use_class

        if use_class:
            num_classes = 10

        # DEPTH ENCODER
        self.conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1).cuda(gpu_device)
        self.conv11d.weight.data = avg

        self.CBR1_D = nn.Sequential(
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats[1].cuda(gpu_device),
            feats[2].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats[3].cuda(gpu_device),
        )
        self.CBR2_D = nn.Sequential(
            feats[5].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats[6].cuda(gpu_device),
            feats[7].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats[8].cuda(gpu_device),
        )
        self.CBR3_D = nn.Sequential(
            feats[10].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats[11].cuda(gpu_device),
            feats[12].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats[13].cuda(gpu_device),
            feats[14].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats[15].cuda(gpu_device),
        )
        self.dropout3_d = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR4_D = nn.Sequential(
            feats[17].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[18].cuda(gpu_device),
            feats[19].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[20].cuda(gpu_device),
            feats[21].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[22].cuda(gpu_device),
        )
        self.dropout4_d = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR5_D = nn.Sequential(
            feats[24].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[25].cuda(gpu_device),
            feats[26].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[27].cuda(gpu_device),
            feats[28].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[29].cuda(gpu_device),
        )

        # RGB ENCODER
        self.CBR1_RGB = nn.Sequential(
            feats2[0].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats2[1].cuda(gpu_device),
            feats2[2].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats2[3].cuda(gpu_device),
        )

        self.CBR2_RGB = nn.Sequential(
            feats2[5].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats2[6].cuda(gpu_device),
            feats2[7].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats2[8].cuda(gpu_device),
        )

        self.CBR3_RGB = nn.Sequential(
            feats2[10].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[11].cuda(gpu_device),
            feats2[12].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[13].cuda(gpu_device),
            feats2[14].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[15].cuda(gpu_device),
        )
        self.dropout3 = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR4_RGB = nn.Sequential(
            feats2[17].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[18].cuda(gpu_device),
            feats2[19].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[20].cuda(gpu_device),
            feats2[21].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[22].cuda(gpu_device),
        )
        self.dropout4 = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR5_RGB = nn.Sequential(
            feats2[24].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[25].cuda(gpu_device),
            feats2[26].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[27].cuda(gpu_device),
            feats2[28].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[29].cuda(gpu_device),
        )
        self.dropout5 = nn.Dropout(p=0.5).cuda(gpu_device)

        if use_class:
            self.ClassHead = nn.Sequential(
                # classifier[0].cuda(gpu_device),
                nn.Linear(35840, 4096).cuda(gpu_device),
                nn.ReLU(),
                nn.Dropout(p=0.5).cuda(gpu_device),
                nn.Linear(4096, 4096).cuda(gpu_device),
                # classifier[3].cuda(gpu_device),
                nn.ReLU(),
                nn.Dropout(p=0.5).cuda(gpu_device),
                nn.Linear(4096, num_classes).cuda(gpu_device)
            )

        # RGB DECODER
        self.CBR5_Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR4_Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR3_Dec = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(256,  128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR2_Dec = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(128, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )

        self.CBR1_Dec = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(64, num_labels, kernel_size=3, padding=1).cuda(gpu_device),
        )

        ## Concat layer
        self.conv2d_1 = nn.Conv2d(128, 64, kernel_size=1, padding=0).cuda(gpu_device)
        self.conv2d_2 = nn.Conv2d(256, 128, kernel_size=1, padding=0).cuda(gpu_device)
        self.conv2d_3 = nn.Conv2d(512, 256, kernel_size=1, padding=0).cuda(gpu_device)
        self.conv2d_4 = nn.Conv2d(1024, 512, kernel_size=1, padding=0).cuda(gpu_device)

        print('[INFO] FuseNet model has been created')
        self.initialize_weights()

    # He Initialization for the linear layers in the classification head
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_out = size[0]  # number of rows
                fan_in = size[1]  # number of columns
                variance = np.sqrt(4.0/(fan_in + fan_out))
                m.weight.data.normal_(0.0, variance)

    def forward(self, rgb_inputs, depth_inputs):


        # DEPTH ENCODER
        # Stage 1
        x = self.conv11d(depth_inputs)
        x_1 = self.CBR1_D(x)
        x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x_2 = self.CBR2_D(x)
        x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x_3 = self.CBR3_D(x)
        x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_D(x)
        x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_D(x)


        # RGB ENCODER
        # Stage 1
        y = self.CBR1_RGB(rgb_inputs)
        # y = torch.add(y, x_1)

        y = torch.cat((y, x_1), 1)
        y = self.conv2d_1(y)  # 1x1 conv

        y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        y = self.CBR2_RGB(y)
        # y = torch.add(y, x_2)

        y = torch.cat((y, x_2), 1)
        y = self.conv2d_2(y)  # 1x1 conv


        y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        y = self.CBR3_RGB(y)
        # y = torch.add(y, x_3)


        y = torch.cat((y, x_3), 1)
        y = self.conv2d_3(y)  # 1x1 conv


        y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB(y)
        # y = torch.add(y,x_4)


        y = torch.cat((y, x_4), 1)
        y = self.conv2d_4(y)  # 1x1 conv



        y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB(y)
        # y = torch.add(y, x_5)

        y = torch.cat((y, x_5), 1)
        y = self.conv2d_4(y)  # 1x1 conv

        y_size = y.size()

        y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout5(y)
        #
        if self.use_class:
            # FC Block for Scene Classification
            y_class = y.view(y.size(0), -1)
            y_class = self.ClassHead(y_class)

        # DECODER
        # Stage 5 dec
        y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
        y = self.CBR5_Dec(y)

        # Stage 4 dec
        y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
        y = self.CBR4_Dec(y)

        # Stage 3 dec
        y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
        y = self.CBR3_Dec(y)

        # Stage 2 dec
        y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
        y = self.CBR2_Dec(y)

        # Stage 1 dec
        y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
        y = self.CBR1_Dec(y)

        if self.use_class:
            return y, y_class
        return y
