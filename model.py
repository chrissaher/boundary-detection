import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable

from enet import Encoder, DecoderModule
from enet import ENCODER_PARAMS, DECODER_PARAMS

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class LossFn:
    """docstring for LossFn."""
    def __init__(self, cls_factor=1, reg_factor=1):
        super(LossFn, self).__init__()
        self.cls_factor = cls_factor
        self.reg_factor = reg_factor
        self.loss_cls = nn.BCELoss()
        self.loss_reg = nn.MSELoss()
        # self.loss_reg = nn.L1Loss()

    def cls_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)

        return self.loss_cls(pred_label, gt_label) * self.cls_factor

    def reg_loss(self, gt_label, gt_offset, pred_offset):
        pref_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        pred_offset = torch.squeeze(pred_offset)

        # unmask = torch.eq(gt_label, 0)
        # mask = torch.eq(unmask, 0)
        mask = torch.eq(gt_label, 1)
        # total = torch.sum(mask)
        # if total == 0:
        #     # print("GT SIZE: ", gt_offset.size())
        #     a = torch.ones((16,2)).float().cuda()
        #     # print("A SIZE: ", a.size())
        #     # return Variable(a)
        #     return self.loss_reg(a, a) * self.reg_factor

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        # print("TOTAL: ", total)
        valid_gt_offset = gt_offset[chose_index, :]
        valid_pred_offset = pred_offset[chose_index, :]
        # try:
        #     print('gt_offset: ', valid_gt_offset)
        #     print('pred_offset: ', valid_pred_offset)
        # except:
        #     print('ERROR')
        return self.loss_reg(valid_pred_offset, valid_gt_offset) * self.reg_factor


class Network(nn.Module):

    def __init__(self, is_train=False, nclasses=4, use_cuda=True):
        super(Network, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.encoder = Encoder(ENCODER_PARAMS, nclasses)


        self.added_layers = nn.Sequential(
            # Custom
            # nn.Conv2d(128, 64, kernel_size=3, stride=1),  # conv1
            # nn.PReLU(),  # prelu1
            # nn.Conv2d(64, 32, kernel_size=3, stride=1),  # conv1
            # nn.PReLU(),  # prelu1
            # nn.MaxPool2d(kernel_size=2,stride=2), # pool3

            # mtcnn
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),  # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )

        # mtcnn
        self.conv5 = nn.Linear(128*17, 256)  # conv5
        # Custom
        # self.conv5 = nn.Linear(32*190, 256)  # conv5
        self.dropout5 = nn.Dropout(0.25)
        self.prelu5 = nn.PReLU()  # prelu5

        self.conv6 = nn.Linear(256, 256)  # conv5
        self.dropout6 = nn.Dropout(0.25)
        self.prelu6 = nn.PReLU()  # prelu5
        # detection
        self.cls_layer = nn.Linear(256, 1)
        # regression
        self.reg_layer = nn.Linear(256, 2)

        self.dual_cls = nn.Linear(256, 2)
        self.dual_reg = nn.Linear(256, 4)
        # weight initiation weih xavier
        # self.apply(weights_init)

    def forward(self, X):

        x = self.added_layers(X)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.dropout5(x)
        x = self.prelu5(x)


        det_1 = torch.sigmoid(self.cls_layer(x))
        det_2 = torch.sigmoid(self.cls_layer(x))
        reg = self.dual_reg(x)

        return det_1, det_2, reg

        # x = self.backend(X)
        #
        # l = self.added_layers(x)
        # l = l.view(l.size(0), -1)
        # l = self.conv5(l)
        # l = self.dropout5(l)
        # l = self.prelu5(l)
        #
        # l = self.conv6(l)
        # l = self.dropout6(l)
        # l = self.prelu6(l)
        #
        # det_l = torch.sigmoid(self.cls_layer(l))
        # reg_l = self.reg_layer(l)
        #
        # r = self.added_layers(x)
        # r = r.view(r.size(0), -1)
        # r = self.conv5(r)
        # r = self.dropout5(r)
        # r = self.prelu5(r)
        #
        # r = self.conv6(r)
        # r = self.dropout6(r)
        # r = self.prelu6(r)
        #
        # det_r = torch.sigmoid(self.cls_layer(r))
        # reg_r = self.reg_layer(r)
        #
        # return det_l, reg_l, det_r, reg_r

# For training 640x240 - Separate Left from Right
class LeftNetwork(nn.Module):
    def __init__(self, is_train=False, nclasses=4, use_cuda=True):
        super(LeftNetwork, self).__init__()

        self.backend = nn.Sequential(
            # mtcnn
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),  # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )

        self.conv5 = nn.Linear(128*85, 256)  # conv5
        # Custom
        self.dropout5 = nn.Dropout(0.25)
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.cls_layer = nn.Linear(256, 1)
        # regression
        self.reg_layer = nn.Linear(256, 2)

        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, X):
        x = self.backend(X)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.dropout5(x)
        x = self.prelu5(x)
        det = torch.sigmoid(self.cls_layer(x))
        reg = self.reg_layer(x)

        return det, reg

# For training 1280x240 - Separate Left from Right
class LeftRightNetworkFullSize(nn.Module):
    def __init__(self, is_train=False, nclasses=4, use_cuda=True):
        super(LeftRightNetworkFullSize, self).__init__()

        self.backend = nn.Sequential(
            # mtcnn
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),  # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )

        self.conv5 = nn.Linear(128*185, 256)  # conv5
        # Custom
        self.dropout5 = nn.Dropout(0.25)
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.cls_layer = nn.Linear(256, 1)
        # regression
        self.reg_layer = nn.Linear(256, 2)

        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, X):
        x = self.backend(X)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.dropout5(x)
        x = self.prelu5(x)
        det = torch.sigmoid(self.cls_layer(x))
        reg = self.reg_layer(x)

        return det, reg


class EncDecFichaNet(nn.Module):

    def __init__(self, is_train=False, nclasses=4, use_cuda=True):
        super(EncDecFichaNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.encoder = Encoder(ENCODER_PARAMS, nclasses)

        self.pooling_modules = []

        for mod in self.encoder.modules():
            try:
                if mod.other.downsample:
                    self.pooling_modules.append(mod.other)
            except AttributeError:
                pass

        self.layers = []
        for i, params in enumerate(DECODER_PARAMS):
            if params['upsample']:
                params['pooling_module'] = self.pooling_modules.pop(-1)
            layer = DecoderModule(**params)
            self.layers.append(layer)
            layer_name = 'decoder{:02d}'.format(i)
            super(EncDecFichaNet,self).__setattr__(layer_name, layer)

        self.added_layers = nn.Sequential(
            # mtcnn
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),  # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )

        # mtcnn
        self.conv5 = nn.Linear(128*185, 256)  # conv5
        # Custom
        # self.conv5 = nn.Linear(32*190, 256)  # conv5
        self.dropout5 = nn.Dropout(0.25)
        self.prelu5 = nn.PReLU()  # prelu5

        self.conv6 = nn.Linear(256, 256)  # conv5
        self.dropout6 = nn.Dropout(0.25)
        self.prelu6 = nn.PReLU()  # prelu5
        # detection
        self.cls_layer = nn.Linear(256, 1)
        # regression
        self.reg_layer = nn.Linear(256, 2)

        self.dual_cls = nn.Linear(256, 2)
        self.dual_reg = nn.Linear(256, 4)
        # weight initiation weih xavier
        # self.apply(weights_init)

    def forward(self, X):
        x = self.encoder(X)
        for layer in self.layers:
            x = layer(x)
        # x = self.added_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.conv5(x)
        # x = self.dropout5(x)
        # x = self.prelu5(x)
        #
        #
        # det_1 = torch.sigmoid(self.cls_layer(x))
        # det_2 = torch.sigmoid(self.cls_layer(x))
        # reg = self.dual_reg(x)
        #
        # return det_1, det_2, reg



        l = self.added_layers(x)
        l = l.view(l.size(0), -1)
        l = self.conv5(l)
        l = self.dropout5(l)
        l = self.prelu5(l)

        l = self.conv6(l)
        l = self.dropout6(l)
        l = self.prelu6(l)

        det_l = torch.sigmoid(self.cls_layer(l))
        reg_l = self.reg_layer(l)

        r = self.added_layers(x)
        r = r.view(r.size(0), -1)
        r = self.conv5(r)
        r = self.dropout5(r)
        r = self.prelu5(r)

        r = self.conv6(r)
        r = self.dropout6(r)
        r = self.prelu6(r)

        det_r = torch.sigmoid(self.cls_layer(r))
        reg_r = self.reg_layer(r)

        return det_l, reg_l, det_r, reg_r


# Full enet trained, only encoder
# Left and Right, separate
class EncoderFichaNet(nn.Module):

    def __init__(self, is_train=False, nclasses=4, use_cuda=True):
        super(EncoderFichaNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.encoder = Encoder(ENCODER_PARAMS, nclasses)

        self.added_layers = nn.Sequential(
            # mtcnn
            nn.Conv2d(128, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),  # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )

        # mtcnn
        self.conv5 = nn.Linear(128*15, 256)  # conv5
        # Custom
        # self.conv5 = nn.Linear(32*190, 256)  # conv5
        self.dropout5 = nn.Dropout(0.25)
        self.prelu5 = nn.PReLU()  # prelu5

        # detection
        self.cls_layer = nn.Linear(256, 1)
        # regression
        self.reg_layer = nn.Linear(256, 2)
        # weight initiation weih xavier
        # self.apply(weights_init)

    def forward(self, X):
        x = self.encoder(X)
        x = self.added_layers(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.dropout5(x)
        x = self.prelu5(x)

        det = torch.sigmoid(self.cls_layer(x))
        reg = self.reg_layer(x)

        return det, reg

# Full enet trained, only encoder
# Left and Right, same encoder, different final layers
# Outputs 20 points (only Y axis)
class EncoderPointsLeftRightFichaNet(nn.Module):

    def __init__(self, is_train=False, nclasses=4, use_cuda=True):
        super(EncoderPointsLeftRightFichaNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.encoder = Encoder(ENCODER_PARAMS, nclasses)

        self.added_layers = nn.Sequential(
            # mtcnn
            nn.Conv2d(128, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),  # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )

        # mtcnn
        # LEFT
        self.conv5 = nn.Linear(128*17, 256)  # conv5
        # Custom
        # self.conv5 = nn.Linear(32*190, 256)  # conv5
        self.dropout5 = nn.Dropout(0.25)
        self.prelu5 = nn.PReLU()  # prelu5

        # detection
        self.cls_layer = nn.Linear(256, 1)
        # regression
        self.reg_layer = nn.Linear(256, 20)

        # RIGHT

        self.conv6 = nn.Linear(128*17, 256)  # conv5
        # Custom
        # self.conv5 = nn.Linear(32*190, 256)  # conv5
        self.dropout6 = nn.Dropout(0.25)
        self.prelu6 = nn.PReLU()  # prelu5

        # detection
        self.cls_layer_2 = nn.Linear(256, 1)
        # regression
        self.reg_layer_2 = nn.Linear(256, 20)
        # weight initiation weih xavier
        # self.apply(weights_init)

    def forward(self, X):
        x = self.encoder(X)

        x = self.added_layers(x)
        l = x.view(x.size(0), -1)
        l = self.conv5(l)
        l = self.dropout5(l)
        l = self.prelu5(l)

        det_l = torch.sigmoid(self.cls_layer(l))
        reg_l = self.reg_layer(l)

        #r = self.added_layers(x)
        r = x.view(x.size(0), -1)
        r = self.conv6(r)
        r = self.dropout6(r)
        r = self.prelu6(r)

        det_r = torch.sigmoid(self.cls_layer_2(r))
        reg_r = self.reg_layer_2(r)

        return det_l, det_r, reg_l, reg_r


class EncoderPointsOneSideFichaNet(nn.Module):

    def __init__(self, is_train=False, nclasses=4, use_cuda=True):
        super(EncoderPointsOneSideFichaNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.encoder = Encoder(ENCODER_PARAMS, nclasses)

        self.added_layers = nn.Sequential(
            # mtcnn
            nn.Conv2d(128, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),  # conv3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.PReLU() # prelu4
        )

        # mtcnn
        self.conv5 = nn.Linear(128*17, 256)  # conv5
        # Custom
        # self.conv5 = nn.Linear(32*190, 256)  # conv5
        self.dropout5 = nn.Dropout(0.25)
        self.prelu5 = nn.PReLU()  # prelu5

        # detection
        self.cls_layer = nn.Linear(256, 1)
        # regression
        self.reg_layer = nn.Linear(256, 20)
        # weight initiation weih xavier
        # self.apply(weights_init)

    def forward(self, X):
        x = self.encoder(X)
        x = self.added_layers(x)

        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.dropout5(x)
        x = self.prelu5(x)

        det = torch.sigmoid(self.cls_layer(x))
        reg = self.reg_layer(x)

        return det, reg
