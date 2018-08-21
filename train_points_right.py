import numpy as np
import cv2
import math
from model import *
from parse_images import get_training_data_right_from_full

import torch
from tqdm import tqdm
from torch.autograd import Variable

from torchsummary import summary

from enet import ENet

from PIL import Image
import torchvision.transforms as standard_transforms

import os

def get_next_batch(gt_images, gt_labels, gt_boxes_r, iter, batch_size, shuffle=True, transform=None):

    start_point = iter * batch_size
    end_point = (iter + 1) * batch_size
    images = []
    boxes_r = []

    X = np.arange(20)
    for gt_image, gt_box_r in zip(gt_images[start_point:end_point], gt_boxes_r[start_point:end_point]):
        box_l = []
        box_r = []
        img = cv2.imread(gt_image)
        pad_size = np.random.randint(51)
        pad_dir = np.random.randint(2) * 2 - 1
        pad_size = 0
        img = np.array(img)
        images.append(img.transpose(2,0,1))

        mr, br = gt_box_r
        if pad_dir == -1:
            for elem in X:
                y = 240 - elem * 5
                x =  (y - br - pad_size) / mr
                box_r.append(x)

        else:
            for elem in X:
                y = 240 - elem * 5
                x =  (y - br + pad_size) / mr
                box_r.append(x)

        boxes_r.append(box_r)

    labels = gt_labels[start_point:end_point]

    images = np.array(images)
    labels = np.array(labels)
    boxes_r = np.array(boxes_r)


    if shuffle == True:
        p = np.random.permutation(len(images))
        images = images[p]
        labels = labels[p]
        boxes_r = boxes_r[p]

    return images, labels, boxes_r

def test_get_data():
    X = np.arange(20)
    color_green = (0,255,0) # green - 128
    color_blue = (255,0, 0) # green - 128
    gt_images, gt_labels, gt_boxes_r = get_training_data_right_from_full(image_path = 'images_complete/', annotation_file='tag.complete.csv', max_w = 1280, max_h = 240)
    images, labels, boxes_r = get_next_batch(gt_images, gt_labels, gt_boxes_r, 250, 5)
    for image, label, box_r in zip(images, labels, boxes_r):
        frame = image.transpose((1,2,0)).astype(np.uint8).copy()

        if label == 1:
            xs = box_r
            for idx, elem in enumerate(X):
                y = 240 - elem * 5
                x = int(xs[idx])
                cv2.circle(img=frame, center=(x,y), radius=5, color=color_blue, thickness=2)

        cv2.imshow('image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def train(epochs = 5000, batch_size=16, lr=0.00001, use_cuda=True):
    print('Epochs: ', epochs)
    print('Batch Size: ', batch_size)
    print('Learning rate: ', lr)
    lossfn = LossFn()
    # net = Network(is_train=True, use_cuda=use_cuda)
    net = EncoderPointsOneSideFichaNet(is_train=True, use_cuda=use_cuda)
    print('Summary')

    # summary(net, (3, 640, 120))
    print('Load pretrained model')
    pretrained_model = ENet(4).cuda()
    state_dict = torch.load('pretrained/ficha_enet.pth')
    pretrained_model.load_state_dict(state_dict)

    net.encoder = pretrained_model.encoder

    freeze_layers = [
        # net.encoder
    ]

    # freeze_layers.extend(net.layers)

    print('Freezing layers')
    for layer in freeze_layers:
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    print('Setting optimizer')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    # transforms = standard_transforms.Compose([
    #     standard_transforms.CenterCrop((240,1280))
    #     standard_transforms.ToTensor()
    # ])

    gt_images, gt_labels, gt_boxes_r = get_training_data_right_from_full(image_path = 'images_complete/', annotation_file='tag.complete.csv', max_w = 1280, max_h = 240)
    train_size = len(gt_images)
    n_iterations = train_size // batch_size
    if n_iterations * batch_size < train_size:
        n_iterations += 1

    print('Size of training set: ', train_size)
    print('Number of iterations per epoch: ', n_iterations)


    net.train()
    if use_cuda:
        net.cuda()

    for epoch in range(epochs):
        avg_tot_loss = 0
        avg_cls_loss = 0
        avg_box_loss = 0

        for iter in tqdm(range(n_iterations)):
            images, labels, boxes_r = get_next_batch(gt_images, gt_labels, gt_boxes_r, iter, batch_size)
            total = np.sum(labels)

            # print("TOTAL: ", total)
            im_tensor = Variable(torch.from_numpy(images).float())
            gt_label = Variable(torch.from_numpy(labels).float())
            # gt_box_l = Variable(torch.from_numpy(boxes_l).float())
            gt_box_r = Variable(torch.from_numpy(boxes_r).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                # gt_box_l = gt_box_l.cuda()
                gt_box_r = gt_box_r.cuda()

            cls_pred, box_pred = net(im_tensor)

            cls_loss = lossfn.cls_loss(gt_label, cls_pred)
            box_loss = lossfn.reg_loss(gt_label, gt_box_r, box_pred, total)

            cls_weight = 1
            reg_weight = 1

            if total == 0:
                all_loss = (cls_loss) * cls_weight
            else:
                all_loss = (cls_loss) * cls_weight
                if math.isnan(box_loss) == False:
                    all_loss += (box_loss * reg_weight)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            avg_tot_loss += all_loss
            avg_cls_loss += (cls_loss)
            if math.isnan(box_loss) == False:
                avg_box_loss += box_loss


        avg_tot_loss = avg_tot_loss * 1.0 / n_iterations
        avg_cls_loss = avg_cls_loss * 1.0 / n_iterations
        avg_box_loss = avg_box_loss * 1.0 / n_iterations
        print("Epoch: %d, tot_loss: %.5f, cls_loss: %.5f, box_loss: %.5f" % (epoch, avg_tot_loss, avg_cls_loss, avg_box_loss))

        if epoch % 10 == 0:
            print("SAVING MODEL")
            torch.save(net, './models/1280x240_points_right/model_epoch_{}.ckpt'.format(epoch))
            print("MODEL SAVED CORRECTLY")
    print("FINISH")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print("#devices: ", torch.cuda.device_count())
    with torch.cuda.device(0):
        print('current device: ', torch.cuda.current_device())
        train()
        # test_get_data()
    # train()
    # test_get_data()
