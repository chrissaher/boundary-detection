import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable


def main():
    files = [
        'car1.mp4',
        'car2.mp4',
        'ped1.mp4',
        'ped2.mp4',
        'ped3.mp4',
        'ped4.mp4',
        'ped5.mp4',
        'ped6.mp4',
        'ped7.mp4',
        'ped8.mp4'
    ]
    for filename in files:
        input_path = os.path.join('video', filename)
        output_path = os.path.join('output', filename.split('.')[0] + "_freeze.mp4")

        video_reader = cv2.VideoCapture(input_path)
        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_writer = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MPEG'),20.0,(frame_w, frame_h))

        net = torch.load('./models/1120x210_enet_mtcnn_pad/model_epoch_300.ckpt')

        nx = 0
        ny = 240
        nw = 1280
        nh = 240
        color = (255,0,0) # blue
        color_64 = (0,0,255) # red - 64
        color_128 = (0,255,0) # green - 128
        for i in tqdm(range(nb_frames)):
            _, frame = video_reader.read()
            image = frame[nx:nx+nw, ny:ny+nh, :]
            image = cv2.resize(image, (1120, 210)).transpose(2,0,1)
            image = np.array([image])

            im_tensor = Variable(torch.from_numpy(image).float()).cuda()
            cls_pred, box_pred = net(im_tensor)

            clsl = cls_pred[0].data.cpu().numpy()
            ml,bl = box_pred[0].data.cpu().tolist()
            # m1,b1,m2,b2 = box_pred[0].data.cpu().tolist()

            # if clsl > 0.5: # Left
            x1 = 0
            y1 = int((ml * x1 + bl) * nh) + ny
            x2 = 0.45
            y2 = int((ml * x2 + bl) * nh) + ny
            x2 = int(x2 * nw)
            cv2.line(img=frame, pt1=(x1,y1), pt2=(x2,y2), color=color_128, thickness=2)
            #
            # if clsr > 0.5: # Right
            #     x1 = 0.55
            #     y1 = int((mr * x1 + br) * nh) + ny
            #     x2 = 1
            #     y2 = int((mr * x2 + br) * nh) + ny
            #     x1 = int(x1 * nw)
            #     x2 = int(x2 * nw)
            #     cv2.line(img=frame, pt1=(x1,y1), pt2=(x2,y2), color=color, thickness=2)

            # cls_pred = cls_pred.data.cpu().numpy()
            # box_pred = box_pred.data.cpu().tolist()
            # if cls_pred[0] > 0.5: # Class is 1
            #     m, b = box_pred[0]
            #     x1 = 0
            #     y1 = int((m * x1 + b) * nh) + ny
            #     x2 = 1
            #     y2 = int((m * x2 + b) * nh) + ny
            #     x2 *= nw
            #     cv2.line(img=frame, pt1=(x1,y1), pt2=(x2,y2), color=color_64, thickness=2)

            video_writer.write(frame)
        video_writer.release()

if __name__ == '__main__':
    main()
