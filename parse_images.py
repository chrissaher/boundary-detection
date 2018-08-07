import os

def get_training_data(image_path='images/', annotation_file='tag.csv', max_w = 1280, max_h = 240):
    images = []
    labels = []
    boxes_l = []
    boxes_r = []
    content = []

    with open(annotation_file, 'r') as f:
        content = f.read().splitlines()

    for line in content:
        data = line.split(',')


        # First tag
        label1 = int(data[1])
        type1 = int(data[2])
        x1 = int(data[3]) * 1.0 / max_w
        y1 = int(data[4]) * 1.0 / max_h
        x2 = int(data[5]) * 1.0 / max_w
        y2 = int(data[6]) * 1.0 / max_h

        m1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 - m1 * x1

        label2 = 0
        type2 = 1 - type1
        m2 = 0
        b2 = 0
        if len(data) > 7:
            label2 = int(data[7])
            type2 = int(data[8])
            x1 = int(data[9]) * 1.0 / max_w
            y1 = int(data[10]) * 1.0 / max_h
            x2 = int(data[11]) * 1.0 / max_w
            y2 = int(data[12]) * 1.0 / max_h

            m2 = (y2 - y1) * 1.0 / (x2 - x1)
            b2 = y1 - m2 * x1


        if type1 == 1 and type2 == 0:
            images.append(os.path.join(image_path, data[0]))
            labels.append([label1, label2])
            boxes_l.append([m1,b1])
            boxes_r.append([m2,b2])
        elif type1 == 0 and type2 == 1:
            images.append(os.path.join(image_path, data[0]))
            labels.append([label2, label1])
            boxes_l.append([m2,b2])
            boxes_r.append([m1,b1])
        else:
            print("### Error in types for image: ", data[0], " type1: ", type1, " type2: ", type2)
            continue


    return images, labels, boxes_l, boxes_r

def get_training_data_mtcnn(image_path='images_mtcnn/', annotation_file='tag.mtcnn.csv', max_w = 1280, max_h = 240):
    images = []
    labels = []
    boxes = []
    content = []

    with open(annotation_file, 'r') as f:
        content = f.read().splitlines()

    for line in content:
        data = line.split(',')


        # First tag
        label1 = int(data[1])
        type1 = int(data[2])
        x1 = int(data[3]) * 1.0 / max_w
        y1 = int(data[4]) * 1.0 / max_h
        x2 = int(data[5]) * 1.0 / max_w
        y2 = int(data[6]) * 1.0 / max_h

        m1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 - m1 * x1

        label2 = 0
        type2 = 1 - type1
        m2 = 0
        b2 = 0
        if len(data) > 7:
            label2 = int(data[7])
            type2 = int(data[8])
            x1 = int(data[9]) * 1.0 / max_w
            y1 = int(data[10]) * 1.0 / max_h
            x2 = int(data[11]) * 1.0 / max_w
            y2 = int(data[12]) * 1.0 / max_h

            m2 = (y2 - y1) * 1.0 / (x2 - x1)
            b2 = y1 - m2 * x1


        if type1 == 1 and type2 == 0:
            images.append(os.path.join(image_path, data[0]))
            labels.append([label1, label2])
            boxes.append([m1,b1, m2,b2])
        elif type1 == 0 and type2 == 1:
            images.append(os.path.join(image_path, data[0]))
            labels.append([label2, label1])
            boxes.append([m2,b2,m1,b1])
        else:
            print("### Error in types for image: ", data[0], " type1: ", type1, " type2: ", type2)
            continue


    return images, labels, boxes

def get_training_data_left(image_path='images_left/', annotation_file='tag.left.csv', max_w = 640, max_h = 240):
    images = []
    labels = []
    boxes = []
    content = []

    with open(annotation_file, 'r') as f:
        content = f.read().splitlines()

    for line in content:
        data = line.split(',')
        if len(data) < 2:
            continue

        # First tag
        label1 = int(data[1])

        type1 = int(data[2])
        x1 = int(data[3]) * 1.0 / max_w
        y1 = int(data[4]) * 1.0 / max_h
        x2 = int(data[5]) * 1.0 / max_w
        y2 = int(data[6]) * 1.0 / max_h

        m1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 - m1 * x1

        images.append(os.path.join(image_path, data[0]))
        labels.append(label1)
        boxes.append([m1,b1])


    return images, labels, boxes

def get_training_data_right(image_path='images_right/', annotation_file='tag.right.csv', max_w = 1280, max_h = 240):
    images = []
    labels = []
    boxes = []
    content = []

    with open(annotation_file, 'r') as f:
        content = f.read().splitlines()

    for line in content:
        data = line.split(',')
        if len(data) < 2:
            continue

        # First tag
        label1 = int(data[1])

        type1 = int(data[2])
        x1 = int(data[3]) * 1.0 / max_w
        y1 = int(data[4]) * 1.0 / max_h
        x2 = int(data[5]) * 1.0 / max_w
        y2 = int(data[6]) * 1.0 / max_h

        m1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 - m1 * x1

        images.append(os.path.join(image_path, data[0]))
        labels.append(label1)
        boxes.append([m1,b1])


    return images, labels, boxes

def get_training_data_rightonly(image_path='images/', annotation_file='tag.csv', max_w = 1280, max_h = 240):
    images = []
    labels = []
    boxes = []
    content = []

    with open(annotation_file, 'r') as f:
        content = f.read().splitlines()

    for line in content:
        data = line.split(',')


        # First tag
        label1 = int(data[1])
        type1 = int(data[2])
        x1 = int(data[3]) * 1.0 / max_w
        y1 = int(data[4]) * 1.0 / max_h
        x2 = int(data[5]) * 1.0 / max_w
        y2 = int(data[6]) * 1.0 / max_h

        m1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 - m1 * x1

        label2 = 0
        type2 = 1 - type1
        m2 = 0
        b2 = 0
        if len(data) > 7:
            label2 = int(data[7])
            type2 = int(data[8])
            x1 = int(data[9]) * 1.0 / max_w
            y1 = int(data[10]) * 1.0 / max_h
            x2 = int(data[11]) * 1.0 / max_w
            y2 = int(data[12]) * 1.0 / max_h

            m2 = (y2 - y1) * 1.0 / (x2 - x1)
            b2 = y1 - m2 * x1


        if type2 == 0:
            images.append(os.path.join(image_path, data[0]))
            labels.append(label2)
            boxes.append([m2,b2])
        elif type1 == 0:
            images.append(os.path.join(image_path, data[0]))
            labels.append(label1)
            boxes.append([m1,b1])
        else:
            print("### Error in types for image: ", data[0], " type1: ", type1, " type2: ", type2)
            continue

    return images, labels, boxes


if __name__ == '__main__':
    images, labels, content = get_training_data_right()
    for i in range(len(images)):
        image = images[i]
        label1 = labels[i]
        m1, b1 = content[i]
        print("Image: %s || lbl1: %d m1: %.8f b1: %.8f "%(image,label1,m1,b1))
        # print("Image: %s Label: %d %.8f * X + %.8f = 0"%(image, label, m,b))
