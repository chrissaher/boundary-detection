import numpy as np
import cv2
import math
from model import *
from parse_images import get_training_data

import torch
from tqdm import tqdm
from torch.autograd import Variable

from torchsummary import summary

from enet import ENet

def get_next_batch(gt_images, gt_labels, gt_boxes_l, gt_boxes_r, iter, batch_size, shuffle=True):

    start_point = iter * batch_size
    end_point = (iter + 1) * batch_size
    # print("SP: %d EP: %d"%(start_point, end_point))
    # print("LI: %d LL: %d LB: %d"%(len(gt_images), len(gt_labels), len(gt_boxes)))
    images = []
    for gt_image in gt_images[start_point:end_point]:
        img = cv2.imread(gt_image)
        img = cv2.resize(img, (640, 120)) # To keep aspect ratio
        images.append(img.transpose(2,0,1))
    labels = gt_labels[start_point:end_point]
    boxes_l = gt_boxes_l[start_point:end_point]
    boxes_r = gt_boxes_r[start_point:end_point]

    images = np.array(images)
    labels = np.array(labels)
    boxes_l = np.array(boxes_l)
    boxes_r = np.array(boxes_r)

    if shuffle == True:
        p = np.random.permutation(len(images))
        images = images[p]
        labels = labels[p]
        boxes_l = boxes_l[p]
        boxes_r = boxes_r[p]

    return images, labels, boxes_l, boxes_r


def train(epochs = 5000, batch_size=16, lr=0.00001, use_cuda=True):
    print('Epochs: ', epochs)
    print('Batch Size: ', batch_size)
    print('Learning rate: ', lr)
    lossfn = LossFn()
    # net = Network(is_train=True, use_cuda=use_cuda)
    net = EncDecFichaNet(is_train=True, use_cuda=use_cuda)
    print('Summary')

    # summary(net, (3, 640, 120))

    print('Load pretrained model')
    pretrained_model = ENet(4).cuda()
    state_dict = torch.load('pretrained/ficha_enet.pth')
    pretrained_model.load_state_dict(state_dict)

    net.backend = pretrained_model.encoder
    # for i in range(len(net.layers)):
    #     net.layers[i] = pretrained_model.decoder.layers[i]
    freeze_layers = [
        net.encoder
    ]

    # freeze_layers.extend(net.layers)

    print('Freezing layers')
    for layer in freeze_layers:
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    print('Setting optimizer')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    gt_images, gt_labels, gt_boxes_l, gt_boxes_r = get_training_data(max_w = 1280, max_h = 240)
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
            images, labels, boxes_l, boxes_r = get_next_batch(gt_images, gt_labels, gt_boxes_l, gt_boxes_r, iter, batch_size)
            total = np.sum(labels)

            im_tensor = Variable(torch.from_numpy(images).float())
            gt_label = Variable(torch.from_numpy(labels).float())
            gt_box_l = Variable(torch.from_numpy(boxes_l).float())
            gt_box_r = Variable(torch.from_numpy(boxes_r).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_box_l = gt_box_l.cuda()
                gt_box_r = gt_box_r.cuda()

            cls_pred_l, box_pred_l, cls_pred_r, box_pred_r = net(im_tensor)

            cls_loss_l = lossfn.cls_loss(gt_label[:,0], cls_pred_l)
            box_loss_l = lossfn.reg_loss(gt_label[:,0], gt_box_l, box_pred_l)

            cls_loss_r = lossfn.cls_loss(gt_label[:,1], cls_pred_r)
            box_loss_r = lossfn.reg_loss(gt_label[:,1], gt_box_r, box_pred_r)

            cls_weight = 1
            reg_weight = 2

            if total == 0:
                all_loss = (cls_loss_l + cls_loss_r) * cls_weight
            else:
                all_loss = (cls_loss_l + cls_loss_r)
                if math.isnan(box_loss_l) == False:
                    all_loss += (box_loss_l * reg_weight)
                if math.isnan(box_loss_r) == False:
                    all_loss += (box_loss_r * reg_weight)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            avg_tot_loss += all_loss
            avg_cls_loss += (cls_loss_l + cls_loss_r)
            if math.isnan(box_loss_l) == False:
                avg_box_loss += box_loss_l
            if math.isnan(box_loss_r) == False:
                avg_box_loss += box_loss_r


        avg_tot_loss = avg_tot_loss * 1.0 / n_iterations
        avg_cls_loss = avg_cls_loss * 1.0 / n_iterations
        avg_box_loss = avg_box_loss * 1.0 / n_iterations
        print("Epoch: %d, tot_loss: %.5f, cls_loss: %.5f, box_loss: %.5f" % (epoch, avg_tot_loss, avg_cls_loss, avg_box_loss))

        if epoch % 50 == 0:
            print("SAVING MODEL")
            torch.save(net, './models/640x120_fullenet/model_epoch_{}.ckpt'.format(epoch))
            print("MODEL SAVED CORRECTLY")
    print("FINISH")

if __name__ == '__main__':
    train()
