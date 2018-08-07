from argparse import ArgumentParser
import os
import cv2
from tqdm import tqdm
import torch
from torch.autograd import Variable
import numpy as np


def parse_args():
    args = ArgumentParser()
    args.add_argument('--video', default='video/car1.mp4', help='path to video for analize')
    args.add_argument('--out', default='logObject.txt', help='file were to store output')
    args.add_argument('--model', default='./models/640x120_fullenet/model_epoch_1000.ckpt', help='model to load')

    return args.parse_args()
def main(args):
    fileName = str(args.video)
    folderName = fileName.split('/')[-1].split('.')[0]
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    f = open(os.path.join(folderName, args.out), 'w')

    video_reader = cv2.VideoCapture(fileName)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    net = torch.load(args.model)

    nx = 0
    ny = 240
    nw = 1280
    nh = 240

    for i in tqdm(range(nb_frames)):
        _, frame = video_reader.read()
        image = frame[nx:nx+nw, ny:ny+nh, :]
        image = cv2.resize(image, (640, 120)).transpose(2,0,1)
        image = np.array([image])

        im_tensor = Variable(torch.from_numpy(image).float()).cuda()
        cls_pred_l, box_pred_l, cls_pred_r, box_pred_r = net(im_tensor)

        clsl = cls_pred_l[0].data.cpu().numpy()
        ml,bl = box_pred_l[0].data.cpu().tolist()
        clsr = cls_pred_r[0].data.cpu().numpy()
        mr,br = box_pred_r[0].data.cpu().tolist()

        nclasses = 0
        left_detected = False
        right_detected = False
        left_arr = []
        right_arr = []

        if clsl > 0.5: # Left
            x1 = 0
            y1 = int((ml * x1 + bl) * nh) + ny
            x2 = 0.45
            y2 = int((ml * x2 + bl) * nh) + ny
            x2 = int(x2 * nw)
            nclasses += 1
            # left_detected = True
            # left_arr = [1, x1, y2, x2 - x1, y1 - y2, 0,0,0,0]
            right_detected = True
            right_arr = [1, x1, y1, x2 - x1, y2 - y1, 0,0,0,0]

        if clsr > 0.5: # Right
            x1 = 0.55
            y1 = int((mr * x1 + br) * nh) + ny
            x2 = 1
            y2 = int((mr * x2 + br) * nh) + ny
            x1 = int(x1 * nw)
            x2 = int(x2 * nw)
            nclasses += 1
            # right_detected = True
            # right_arr = [1, x1, y1, x2 - x1, y2 - y1, 0,0,0,0]
            left_detected = True
            left_arr = [1, x1, y2, x2 - x1, y1 - y2, 0,0,0,0]


        frameName = '{}.jpg'.format(i)

        line = frameName + ' ' + str(nclasses) + ' '
        if left_detected:
            for elem in left_arr:
                line = line + str(elem) + ' '

        if right_detected:
            for elem in right_arr:
                line = line + str(elem) + ' '

        line += '\n'
        f.write(line)
        cv2.imwrite(os.path.join(folderName, frameName),frame)

    f.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
