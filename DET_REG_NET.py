import os
import cv2
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import m0_openslide_read_region as slide

from PIL import Image
from scipy.io import loadmat, savemat
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

lr = 0.001
num_epochs = 200
batch_size = 2


class det_cls_tr(Dataset):
    def __init__(self, root='./data/CRC/Detection', use_H=True, random_crop=False, crop_size=100):
        super(det_cls_tr, self).__init__()
        self.root = root
        self.use_H = use_H
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.list = [one for one in sorted(os.listdir(self.root)) if one.startswith('img')]
        self.transform = transforms.RandomCrop(size=256)

    def __getitem__(self, item):
        if not self.use_H:
            img = Image.open(os.path.join(self.root, self.list[item], self.list[item] + ".bmp"))
        else:
            img = Image.open(os.path.join(self.root, self.list[item], self.list[item] + "_H.png"))
        detection = loadmat(os.path.join(self.root, self.list[item], self.list[item] + "_detection.mat"))['detection']

        detection_map = np.zeros(img.size, dtype=float)
        for x, y in detection:
            x, y = int(x) - 1, int(y) - 1
            detection_map[y][x] = 100.0
            for i in range(x - 5, x + 6):
                for j in range(y - 5, y + 6):
                    if i < 0 or j < 0 or i >= 500 or j >= 500:
                        continue
                    dist_x, dist_y = abs(x - i), abs(y - j)
                    dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
                    if dist <= 5:
                        pix_value = 1.0 / (1.0 + (dist ** 2) * 0.5)
                        assert pix_value <= 1
                        if detection_map[j][i] != 0 and pix_value * 100.0 > detection_map[j][i]:
                            detection_map[j][i] = pix_value * 100.0
                        if detection_map[j][i] == 0:
                            detection_map[j][i] = pix_value * 100.0

        w, h = img.size
        img = np.array(img)
        if self.random_crop:
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
            img = img[x1: x1 + self.crop_size, y1: y1 + self.crop_size, :]
            detection_map = detection_map[x1: x1 + self.crop_size, y1: y1 + self.crop_size]

            assert img.shape[0] == img.shape[1] and img.shape[0] == detection_map.shape[0]
            assert detection_map.shape[0] == detection_map.shape[1]
            assert img.shape[0] == self.crop_size
        # plt.imsave('/home/xjw/Projects/tmi2016_ks/data/CRC/label/' + self.list[item], detection_map)
        # savemat('/home/xjw/Projects/tmi2016_ks/data/CRC/label/' + self.list[item], {'det': detection_map})
        return np.transpose(img, (2, 0, 1)), np.expand_dims(detection_map, axis=0), self.list[item]

    def __len__(self):
        return len(self.list)


class det_cls_te(Dataset):
    def __init__(self, root='./data/CRC/test_data', use_H=False):
        super(det_cls_te, self).__init__()
        if not use_H:
            self.root = root
        else:
            self.root = os.path.join(root, 'H')

        self.list = []
        for root, _, file_names in os.walk(self.root):
            for file_name in file_names:
                if file_name.startswith('img'):
                    if not use_H and 'H' not in file_name and 'E' not in file_name:
                        self.list.append(os.path.join(root, file_name))
                    elif use_H and 'H' in file_name:
                        self.list.append(os.path.join(root, file_name))
                    else: continue
        # self.list = [one for one in sorted(os.listdir(self.root)) if one.startswith('img_test')]

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.list[item]))

        # plt.imsave('/home/xjw/Projects/tmi2016_ks/data/CRC/label/' + self.list[item], detection_map)
        return np.transpose(np.array(img), (2, 0, 1)), self.list[item]

    def __len__(self):
        return len(self.list)


class ce_loss(nn.Module):
    """
    MUST BE WRONG !!!
    """
    def __init__(self):
        super(ce_loss, self).__init__()

    def forward(self, inputs, target):
        loss = 0.
        inputs = torch.sigmoid(inputs)
        target = 1. * target / 100.
        n = inputs.size(0)
        for i in range(n):
            pix_p, pix_l = inputs[i, :, :, :].view(-1), target[i, :, :, :].view(-1)
            loss += torch.sum((pix_l + 1e-8) * (-1. * (pix_l * torch.log(pix_p) + (1. - pix_l) * torch.log(1. - pix_p))))
        return loss / (n * 1.)


class DET_REG_NET(nn.Module):
    def __init__(self, num_channels=3, size=500):
        super(DET_REG_NET, self).__init__()
        self.feature_extraction = nn.Sequential(nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2),
                                                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                                nn.BatchNorm2d(64),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2),
                                                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                                nn.BatchNorm2d(128),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2),
                                                nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
                                                nn.BatchNorm2d(512),
                                                nn.ReLU(), )
        self.upsampling = nn.Sequential(nn.UpsamplingBilinear2d(size=(size // 2 ** 2, size // 2 ** 2)),
                                        nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.UpsamplingBilinear2d(size=(size // 2, size // 2)),
                                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.UpsamplingBilinear2d(size=(size, size)),
                                        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1), )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.upsampling(x)
        return x


class DET_REG_NET_v2(nn.Module):
    """
    Use VGG to replace the basic downsampling pipeline
    """
    def __init__(self, backbone='vgg16', pretrained_base=True, size=500, **kwargs):
        super(DET_REG_NET_v2, self).__init__()
        if backbone == 'vgg16':
            self.pretrained = torchvision.models.vgg16_bn(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.upsampling = nn.Sequential(nn.UpsamplingBilinear2d(size=(size // 2 ** 4, size // 2 ** 4)),
                                        nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.UpsamplingBilinear2d(size=(size // 2 ** 3, size // 2 ** 3)),
                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.UpsamplingBilinear2d(size=(size // 2 ** 2, size // 2 ** 2)),
                                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.UpsamplingBilinear2d(size=(size // 2, size // 2)),
                                        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.UpsamplingBilinear2d(size=(size, size)),
                                        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),)

    def forward(self, x):
        x = self.pretrained(x)
        x = self.upsampling(x)

        return x


class DET_REG_NET_v3(nn.Module):
    """
        Follow the U-Net structure
        """

    def __init__(self, backbone='vgg16', pretrained_base=True, num_channels=3, size=500, **kwargs):
        super(DET_REG_NET_v3, self).__init__()

        if backbone == 'vgg16':
            self.block1 = nn.Sequential(
                nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
            self.block4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
            self.block5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.up1 = nn.Sequential(nn.UpsamplingBilinear2d(size=(size // 2 ** 4, size // 2 ** 4)),
                                 nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(), )
        self.up2 = nn.Sequential(nn.UpsamplingBilinear2d(size=(size // 2 ** 3, size // 2 ** 3)),
                                 nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(), )
        self.up3 = nn.Sequential(nn.UpsamplingBilinear2d(size=(size // 2 ** 2, size // 2 ** 2)),
                                 nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(), )
        self.up4 = nn.Sequential(nn.UpsamplingBilinear2d(size=(size // 2, size // 2)),
                                 nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(), )
        self.up5 = nn.Sequential(nn.UpsamplingBilinear2d(size=(size, size)),
                                 nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1), )

    #     if pretrained_base:
    #         self.pretrain = self.load_pretrain(self.downsampling)
    #
    # def load_pretrain(self, model):
    #     from torch.hub import load_state_dict_from_url
    #     state_dict = load_state_dict_from_url(
    #         'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', progress=True)
    #     model.load_state_dict(state_dict)
    #     return model

    def base_forward(self, x):
        c1 = self.block1(x) # 256
        c2 = self.block2(c1) # 128
        c3 = self.block3(c2) # 64
        c4 = self.block4(c3) # 32
        c5 = self.block5(c4) # 15

        return c1, c2, c3, c4, c5

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.base_forward(x)

        x_up1 = self.up1(c5)
        x_up2 = self.up2(torch.cat([x_up1, c4], 1))
        x_up3 = self.up3(torch.cat([x_up2, c3], 1))
        x_up4 = self.up4(torch.cat([x_up3, c2], 1))
        x_up5 = self.up5(torch.cat([x_up4, c1], 1))

        return x_up5


def get_jet():
    from matplotlib import cm
    cmap = np.zeros((256, 3), np.uint8)
    for i in range(256):
        for c in range(3):
            cmap[i, c] = np.int(np.round(cm.jet(i)[c] * 255.0))
    return cmap


def normlize_255(data):
    return np.round(((data - np.min(data)) / (np.max(data) - np.min(data))) * 255).astype(np.uint8)


def to_jet_cmap(path, output):
    cmap = get_jet()
    output_cmap = np.zeros((output.shape[0], output.shape[1], 3), np.uint8)
    output = normlize_255(output)

    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            output_cmap[r, c, :] = cmap[output[r, c]]
    Image.fromarray(output_cmap).save(path)


def creat_nms_boxes(data, kernel_size=20):
    data = normlize_255(data)

    k = kernel_size // 2
    bounding_boxes, confidence = [], []
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if data[r, c] > 25:
                top_left = \
                    [r - k if r - k >= 0 else 0] + \
                    [c - k if c - k >= 0 else 0]
                bottom_right = \
                    [r + k if r + k <= data.shape[0] else data.shape[0]] + \
                    [c + k if c + k <= data.shape[1] else data.shape[1]]
                bounding_boxes.append(top_left + bottom_right)
                confidence.append(data[r, c] / 255.)
    return np.array(bounding_boxes), np.array(confidence)


def nms(boxes, confidence, thresh=0.5):
    picked_boxes, picked_score = [], []

    top_left_x = boxes[:, 0]
    top_left_y = boxes[:, 1]
    bottom_right_x = boxes[:, 2]
    bottom_right_y = boxes[:, 3]

    area = (bottom_right_x - top_left_x + 1) * (bottom_right_y - top_left_y + 1)
    order = np.argsort(confidence)

    while order.size > 0:
        index = order[-1]

        picked_boxes.append(boxes[index])
        picked_score.append(confidence[index])

        x1 = np.maximum(top_left_x[index], top_left_x[order[:-1]])
        x2 = np.minimum(bottom_right_x[index], bottom_right_x[order[:-1]])
        y1 = np.maximum(top_left_y[index], top_left_y[order[:-1]])
        y2 = np.minimum(bottom_right_y[index], bottom_right_y[order[:-1]])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        mIOU = intersection / (area[index] + area[order[:-1]] - intersection)
        order = order[np.where(mIOU < thresh)]
    return picked_boxes, picked_score


def post_processing_nms(data, image, name):
    image = np.transpose(np.squeeze(image), [1, 2, 0]).copy()
    boxes, confidence = creat_nms_boxes(data)
    picked_boxes, picked_score = nms(boxes, confidence)
    for (sx, sy, ex, ey), c in zip(picked_boxes, picked_score):
        cv2.rectangle(image, (sx, sy), (ex, ey), 255)
    plt.imsave(name, image)


def post_processing_thresh(data, image, name, thresh=0.4, with_cls=False, model_cls=None):
    if with_cls and model_cls is None:
        raise AttributeError("Need classification model ... ")

    start, end = [], []
    data = normlize_255(data)
    image = np.transpose(np.squeeze(image), [1, 2, 0]).copy()
    data = (data > thresh * 255).astype(np.uint8) * 255
    contour, _ = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for one in contour:
        [x, y] = np.round(np.mean(np.squeeze(one), axis=0)) if one.shape[0] != 1 else np.round(np.squeeze(one))
        sx = x - 5 if x - 5 >= 0 else 0
        sy = y - 5 if y - 5 >= 0 else 0
        ex = x + 5 if x + 5 < data.shape[0] else data.shape[0] - 1
        ey = y + 5 if y + 5 < data.shape[1] else data.shape[1] - 1
        start.append((sx, sy))
        end.append((ex, ey))

    # make prediction on classes
    class_information, patch_zip = [], []
    for (sx, sy), (ex, ey) in zip(start, end):
        sx, sy, ex, ey = list(map(int, (sx, sy, ex, ey)))

        sx_c = sx - 11 if sx - 11 >= 0 else 0
        sy_c = sy - 11 if sy - 11 >= 0 else 0
        ex_c = ex + 11 if ex + 11 < data.shape[0] else data.shape[0] - 1
        ey_c = ey + 11 if ey + 11 < data.shape[1] else data.shape[1] - 1

        ex_c = 32 if sx_c == 0 else ex_c
        ey_c = 32 if sy_c == 0 else ey_c
        sx_c = 467 if ex_c == 499 else sx_c
        sy_c = 467 if ey_c == 499 else sy_c

        patch_zip.append(image[sy_c: ey_c, sx_c: ex_c, :])
        assert patch_zip[-1].shape == (32, 32, 3)

    patch_zip = torch.from_numpy(np.transpose(np.array(patch_zip, dtype=np.uint8), (0, 3, 1, 2)))
    cls_output = torch.argmax(model_cls(patch_zip.type(torch.FloatTensor).to(device)), dim=1).cpu().numpy()

    # creat rectangle with color
    cmap = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (128, 128, 128)}
    for predict, (sx, sy), (ex, ey) in zip(cls_output, start, end):
        sx, sy, ex, ey = int(sx), int(sy), int(ex), int(ey)
        cv2.rectangle(image, (sx, sy), (ex, ey), cmap[predict])
    plt.imsave(name, image)


def check_folder(p_name, checkpoints_p_name):
    base_dir = os.path.curdir

    total_dirs = {
        "check_points_dir": os.path.join(base_dir, 'checkpoints', checkpoints_p_name),
        "predict_mat_dir": os.path.join(base_dir, 'predict', p_name, 'mat'),
        "predict_nms_dir": os.path.join(base_dir, 'predict', p_name, 'nms'),
        "predict_thresh_dir": os.path.join(base_dir, 'predict', p_name, 'thresh'),
        "predict_jet256_dir": os.path.join(base_dir, 'predict', p_name, 'jet256'),
        "predict_direct_dir": os.path.join(base_dir, 'predict', p_name, 'direct'),
    }
    for one_dir in total_dirs.values():
        if not os.path.exists(one_dir):
            os.makedirs(one_dir)

    return total_dirs


def single_stage_train(p_name, useH):
    folder_dict = check_folder(p_name, checkpoints_p_name=p_name)
    train_dataset = det_cls_tr(use_H=useH)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = DET_REG_NET_v3(num_channels=3).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        forward_loss = []
        for i, (img, lb, _) in enumerate(train_loader):
            images = img.type(torch.FloatTensor).to(device)
            labels = lb.type(torch.FloatTensor).to(device)

            output = model(images)
            loss = criterion(output, labels)
            forward_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch: [{}/{}], step: [{}/{}], loss: {}'.format(
                epoch + 1, num_epochs, i + 1, total_step, np.mean(forward_loss)))

        torch.save(model.state_dict(), os.path.join(
            folder_dict["check_points_dir"], p_name + "_net_params_epoch" + str(epoch) + ".pkl"))


def inference(p_name, useH):
    folder_dict = check_folder(p_name, checkpoints_p_name=p_name)
    test_dataset = det_cls_te(use_H=useH)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model_reg = DET_REG_NET_v3()
    model_reg.load_state_dict(torch.load(os.path.join(
                folder_dict['check_points_dir'], p_name + '_net_params_epoch99.pkl')))
    model_reg.to(device)
    model_reg.eval()

    model_cls = get_classification_model(eff_type=0, num_classes=4)
    model_cls.load_state_dict(torch.load('./checkpoints/classification/EfficientNet-b0-49.pkl'))
    model_cls.to(device)
    model_cls.eval()

    print('Creat label with {}'.format({
        'epithelial': (255, 0, 0),
        'fibroblast': (0, 255, 0),
        'inflammatory': (0, 0, 255),
        'others': (128, 128, 128)}))

    with torch.no_grad():
        for i, (img, name) in enumerate(test_loader):
            name = tuple([name[0].split('/')[-1]], )
            images = img.type(torch.FloatTensor).to(device)

            output = np.squeeze(model_reg(images).cpu().numpy())
            img = img.cpu().numpy()

            # post_processing_nms(output, img, os.path.join(
            #     folder_dict['predict_nms_dir'], "test_" + name[0].replace('bmp', 'tif')))

            post_processing_thresh(output, img, os.path.join(
                folder_dict['predict_thresh_dir'], "test_" + name[0].replace('bmp', 'tif')),
                thresh=0.3, with_cls=True, model_cls=model_cls)

            # plt.imsave(os.path.join(
            #     folder_dict['predict_direct_dir'], "test_" + name[0].replace('bmp', 'tif')), output)
            #
            # savemat(os.path.join(
            #     folder_dict['predict_mat_dir'], "test_" + name[0].replace('bmp', 'mat')), {"img": output})
            #
            # to_jet_cmap(os.path.join(
            #     folder_dict['predict_jet256_dir'], "test_" + name[0].replace('bmp', 'tif')), output)


def inference_WSI(p_name, useH, checkpoints_p_name, do_unpack=False):
    if do_unpack:
        # unpack WSI to patches
        slide.check_python_version()
        slidepath = '/home/xjw/Projects/tmi2016_ks/data/WSI/ndpi/'
        outPath = '/home/xjw/Projects/tmi2016_ks/data/WSI/patches/'

        loader = slide.SlideLoader(slidepath, outPath, downsample=1, endsFormat='ndpi',
                                   patchsize=500, nProcs=4, overlap=0, default_overlap_size=300,
                                   removeBlankPatch=1, blankRange=[190, 210])
        loader.mainProcess(skip=False, nskip=None)

    # start inferencing
    folder_dict = check_folder(p_name, checkpoints_p_name=checkpoints_p_name)
    test_dataset = det_cls_te(root='./data/WSI/patches/img_test_NGH-1539946-01_102_71', use_H=useH)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = DET_REG_NET_v3()
    model.load_state_dict(torch.load(os.path.join(
        folder_dict['check_points_dir'], checkpoints_p_name + '_net_params_epoch99.pkl')))
    model.to(device)

    with torch.no_grad():
        for i, (img, name) in enumerate(test_loader):
            assert test_loader.batch_size == 1
            name = tuple([name[0].split('/')[-1]], )
            images = img.type(torch.FloatTensor).to(device)

            output = np.squeeze(model(images).cpu().numpy())
            img = img.cpu().numpy()

            # post_processing_nms(output, img, os.path.join(
            #     folder_dict['predict_nms_dir'], "test_" + name[0].replace('bmp', 'tif')))

            post_processing_thresh(output, img, os.path.join(
                folder_dict['predict_thresh_dir'], "test_" + name[0].replace('bmp', 'tif')), thresh=0.3)

            plt.imsave(os.path.join(
                folder_dict['predict_direct_dir'], "test_" + name[0].replace('bmp', 'tif')), output)

            savemat(os.path.join(
                folder_dict['predict_mat_dir'], "test_" + name[0].replace('bmp', 'mat')), {"img": output})

            to_jet_cmap(os.path.join(
                folder_dict['predict_jet256_dir'], "test_" + name[0].replace('bmp', 'tif')), output)


def multi_stage_train(p_name, useH):
    folder_dict = check_folder(p_name, checkpoints_p_name=p_name)

    train_dataset_crop = det_cls_tr(use_H=useH, random_crop=True, crop_size=256)
    train_loader_crop = DataLoader(dataset=train_dataset_crop, batch_size=batch_size, shuffle=True)

    # stage1 : shape=256x256
    model = DET_REG_NET_v3(num_channels=3, size=256).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_step = len(train_loader_crop)
    for epoch in range(int(num_epochs * 0.6)):
        forward_loss = []
        for i, (img, lb, _) in enumerate(train_loader_crop):
            images = img.type(torch.FloatTensor).to(device)
            labels = lb.type(torch.FloatTensor).to(device)

            output = model(images)
            loss = criterion(output, labels)
            forward_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Stage: [0/1],  Epoch: [{}/{}], step: [{}/{}], loss: {}'.format(
                epoch + 1, num_epochs, i + 1, total_step, np.mean(forward_loss)))

        torch.save(model.state_dict(), os.path.join(
            folder_dict["check_points_dir"], p_name + "_net_params_epoch" + str(epoch) + ".pkl"))

    # stage2: shape=500x500
    train_dataset = det_cls_tr(use_H=useH, random_crop=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = DET_REG_NET_v3(num_channels=3, size=500)
    model.load_state_dict(torch.load(os.path.join(
        folder_dict['check_points_dir'], p_name + '_net_params_epoch' + str(int(num_epochs * 0.6) - 1) + '.pkl')))
    model.to(device)

    total_step = len(train_loader)
    for epoch in range(int(num_epochs * 0.4)):
        forward_loss = []
        for i, (img, lb, _) in enumerate(train_loader):
            images = img.type(torch.FloatTensor).to(device)
            labels = lb.type(torch.FloatTensor).to(device)

            output = model(images)
            loss = criterion(output, labels)
            forward_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Stage: [2/2], Epoch: [{}/{}], step: [{}/{}], loss: {}'.format(
                epoch + 1, num_epochs, i + 1, total_step, np.mean(forward_loss)))

        torch.save(model.state_dict(), os.path.join(
            folder_dict["check_points_dir"], p_name + "_net_params_epoch" + str(epoch + int(num_epochs * 0.6)) + ".pkl"))


def train(p_name, multi_stage=False, useH=False):
    if multi_stage:
        multi_stage_train(p_name, useH=useH)
    else:
        single_stage_train(p_name, useH=useH)


def puzzle_WSI(sz=125):
    basic_root = './predict/UNet_VGG16_WSI'
    WSI_path = './data/WSI/patches'

    jet_path = os.path.join(basic_root, 'jet256')
    direct_path = os.path.join(basic_root, 'direct')
    thresh_path = os.path.join(basic_root, 'thresh')

    out_path = os.path.join(basic_root, 'WSI_output')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    slide_name = "_".join(['test'] + os.listdir(WSI_path)[0].split('_')[:-2])
    w, h = list(map(int, os.listdir(WSI_path)[0].split('_')[-2:]))

    for real_path in [jet_path, direct_path, thresh_path]:
        for i in range(w):
            for j in range(h):
                if type != 'thresh_path': patch_name = slide_name + '_' + str(i) + '_' + str(j) + '.jpg'
                else: patch_name = slide_name + '_' + str(i) + '_' + str(j) + '.png'
                try:
                    patch = Image.open(os.path.join(real_path, patch_name)).resize((sz, sz))
                    patch = np.array(patch)
                except FileNotFoundError:
                    patch = np.zeros((sz, sz, 3))
                if j == 0:
                    column = patch
                else:
                    column = np.vstack((column, patch))
            if i == 0:
                puzzle = column
            else:
                puzzle = np.hstack((puzzle, column))
        cv2.imwrite(os.path.join(out_path, real_path.split('/')[-1] + '.png'), puzzle)


class MyClassificationModel(nn.Module):
    def __init__(self, input_size=41, num_classes=3):
        super(MyClassificationModel, self).__init__()
        self.feature = torchvision.models.vgg11_bn(pretrained=True).features
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(in_features=256*1*1, out_features=128),
                                        nn.ReLU(),
                                        nn.Linear(in_features=128, out_features=num_classes), )
        p = 10

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x


def get_classification_model(eff_type=0, num_classes=4):
    model = EfficientNet.from_pretrained("efficientnet-b" + str(eff_type), num_classes=num_classes)
    return model


class ClsDataset_tr(Dataset):
    def __init__(self, root='./data/CRC/Classification', num_classes=4, creat_patch=True):
        super(ClsDataset_tr, self).__init__()
        self.root = root
        self.num_classes = num_classes
        self.img_list = [one for one in sorted(os.listdir(self.root)) if one.startswith('img')]
        if creat_patch:
            self.creat_save_patch()
        self.patch_list = []
        for root, _, file_names in os.walk(self.root):
            for file_name in file_names:
                if file_name.endswith('jpg'):
                    self.patch_list.append(os.path.join(root, file_name))
        self.label_list = ('epithelial', 'fibroblast', 'inflammatory', 'others')

    def creat_save_patch(self, patch_size=32):
        for item in range(len(self.img_list)):
            img = Image.open(os.path.join(self.root, self.img_list[item], self.img_list[item] + ".bmp"))
            cls_label_list = [one for one in sorted(os.listdir(os.path.join(
                self.root, self.img_list[item]))) if one.endswith('mat')]

            detection = {}
            for i in range(self.num_classes):
                detection[cls_label_list[i].replace('.mat', '')] = loadmat(
                    os.path.join(self.root, self.img_list[item], cls_label_list[i]))['detection']

            img_path = os.path.join(self.root, self.img_list[item], 'patch')
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            half_size = patch_size // 2 - 1
            for key in detection.keys():
                for idx, (x, y) in enumerate(zip(list(map(int, detection[key][:, 0])),
                                                 list(map(int, detection[key][:, 1])))):
                    left = x - half_size if x - half_size >= 0 else 0
                    up = y - half_size if y - half_size >= 0 else 0
                    left = left if left + patch_size < 500 else 500 - patch_size - 1
                    up = up if up + patch_size < 500 else 500 - patch_size - 1

                    patch = img.crop((left, up, left + patch_size, up + patch_size))
                    patch.save(os.path.join(
                        img_path, '_'.join([self.img_list[item], key.split('_')[-1], str(idx) + '.jpg'])))

    def __getitem__(self, item):
        patch = np.array(Image.open(self.patch_list[item]))
        label = self.label_list.index(self.patch_list[item].split('_')[1])

        return np.transpose(patch, (2, 0, 1)), label, self.patch_list[item]

    def __len__(self):
        return len(self.patch_list)


def classification_train(epochs, cls_batch_size):
    train_dataset = ClsDataset_tr(creat_patch=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cls_batch_size, shuffle=True)

    model = get_classification_model(eff_type=0, num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_step = len(train_loader)
    for epoch in range(epochs):
        forward_loss = []
        for i, (img, lb, _) in enumerate(train_loader):
            images = img.type(torch.FloatTensor).to(device)
            labels = lb.type(torch.LongTensor).to(device)

            output = model(images)
            loss = criterion(output, labels)
            forward_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch: [{}/{}], step: [{}/{}], loss: {}'.format(
                epoch + 1, epochs, i + 1, total_step, np.mean(forward_loss)))

        torch.save(model.state_dict(), './checkpoints/classification/EfficientNet-b0-' + str(epoch) + '.pkl')


def classification_eval(cls_batch_size=1):
    train_dataset = ClsDataset_tr(creat_patch=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cls_batch_size, shuffle=True)

    model = get_classification_model(eff_type=0, num_classes=4)
    model.load_state_dict(torch.load('./checkpoints/classification/EfficientNet-b0-49.pkl'))
    model.to(device)
    model.eval()

    correct, wrong = 0, 0
    with torch.no_grad():
        for i, (img, lb, _) in enumerate(train_loader):
            output = model(img.type(torch.FloatTensor).to(device))
            output = torch.argmax(output, dim=1)

            if output.item() == lb.item():
                correct += 1
            else:
                wrong += 1
        if i % 100 == 0:
            print('ID: {}, Current accuracy: {}'.format(i, 1. * correct / (1. * correct + 1. * wrong)))


if __name__ == '__main__':
    # train(p_name="UNet_VGG16", multi_stage=False, useH=False)
    # inference_WSI(p_name='UNet_VGG16_WSI', useH=False, checkpoints_p_name='UNet_VGG16')
    # classification_train(epochs=50, cls_batch_size=16)
    # classification_eval(cls_batch_size=1) # Accuracy: 0.9930939226519337
    inference(p_name='UNet_VGG16', useH=False)
    # puzzle_WSI()