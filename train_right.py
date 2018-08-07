import numpy as np
import cv2
import math
from model import *
from parse_images import get_training_data_right

import torch
from tqdm import tqdm
from torch.autograd import Variable

from torchsummary import summary

from enet import ENet

def get_next_batch(gt_images, gt_labels, gt_boxes, iter, batch_size, shuffle=True):

    start_point = iter * batch_size
    end_point = (iter + 1) * batch_size
    images = []
    for gt_image in gt_images[start_point:end_point]:
        img = cv2.imread(gt_image)
        img = cv2.resize(img, (320, 60)) # To keep aspect ratio
        images.append(img.transpose(2,0,1))
    labels = gt_labels[start_point:end_point]
    boxes = gt_boxes[start_point:end_point]

    images = np.array(images)
    labels = np.array(labels)
    boxes = np.array(boxes)

    if shuffle == True:
        p = np.random.permutation(len(images))
        images = images[p]
        labels = labels[p]
        boxes = boxes[p]

    return images, labels, boxes


def train(epochs = 5000, batch_size=16, lr=0.00001, use_cuda=True):
    print('Epochs: ', epochs)
    print('Batch Size: ', batch_size)
    print('Learning rate: ', lr)
    lossfn = LossFn()
    # net = Network(is_train=True, use_cuda=use_cuda)
    net = LeftRightNetworkFullSize(is_train=True, use_cuda=use_cuda)
    print('Summary')

    # summary(net, (3, 640, 120))

    print('Setting optimizer')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    gt_images, gt_labels, gt_boxes = get_training_data_right(max_w = 1280, max_h = 240)
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
            images, labels, boxes = get_next_batch(gt_images, gt_labels, gt_boxes, iter, batch_size)
            total = np.sum(labels)

            im_tensor = Variable(torch.from_numpy(images).float())
            gt_label = Variable(torch.from_numpy(labels).float())
            gt_box = Variable(torch.from_numpy(boxes).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_box = gt_box.cuda()

            cls_pred, box_pred = net(im_tensor)

            cls_loss = lossfn.cls_loss(gt_label, cls_pred)
            box_loss = lossfn.reg_loss(gt_label, gt_box, box_pred)

            cls_weight = 1
            reg_weight = 2

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

        if epoch % 50 == 0:
            print("SAVING MODEL")
            torch.save(net, './models/320x60_right/model_epoch_{}.ckpt'.format(epoch))
            print("MODEL SAVED CORRECTLY")
    print("FINISH")

if __name__ == '__main__':
    train()
