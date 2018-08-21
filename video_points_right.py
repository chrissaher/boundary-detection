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
        output_path = os.path.join('output', filename.split('.')[0] + "_points_right.mp4")

        video_reader = cv2.VideoCapture(input_path)
        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_writer = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MPEG'),20.0,(frame_w, frame_h))

        net = torch.load('./models/1280x240_points_right/model_epoch_830.ckpt')

        nx = 0
        ny = 240
        nw = 1280
        nh = 240
        color_blue = (255,0,0) # blue
        color_red = (0,0,255) # red - 64
        color_green = (0,255,0) # green - 128
        X = np.arange(20)
        for i in tqdm(range(nb_frames)):
            _, frame = video_reader.read()
            image = frame[ny:ny+nh, nx:nx+nw, :]
            image = image.transpose(2,0,1)
            image = np.array([image])
            # print(image.shape)

            im_tensor = Variable(torch.from_numpy(image).float()).cuda()
            cls_pred_r, box_pred_r = net(im_tensor)

            lblr = cls_pred_r[0].data.cpu().numpy()
            box_r = box_pred_r[0].data.cpu().tolist()
            # m1,b1,m2,b2 = box_pred[0].data.cpu().tolist()

            if lblr == 1:
                xs = box_r
                for idx, elem in enumerate(X):
                    y = 240 - elem * 5
                    x = int(xs[idx])
                    y = int(y + 240)
                    cv2.circle(img=frame, center=(x,y), radius=5, color=color_blue, thickness=2)


            video_writer.write(frame)
        video_writer.release()

if __name__ == '__main__':
    main()
