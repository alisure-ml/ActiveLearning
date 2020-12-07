import os
import math
import torch
import random
import platform
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from alisuretool.Tools import Tools
from torch.optim import lr_scheduler
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


##############################################################################################################


class CNNEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        pass

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class RelationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 3 * 3, 8)
        self.fc2 = nn.Linear(8, 1)
        pass

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out = out2.view(out2.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


# Original --> MaxPool, FC input
class CNNEncoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        pass

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class RelationNetwork1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 5 * 5, 64)  # 64
        self.fc2 = nn.Linear(64, 1)  # 64
        pass

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out = out2.view(out2.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


##############################################################################################################


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=-1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ICModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(in_dim, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(in_dim, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out_logits = self.linear(out)
        out_l2norm = self.l2norm(out_logits)
        return out_logits, out_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ICResNet(nn.Module):

    def __init__(self, resnet, low_dim=512):
        super().__init__()
        self.resnet = resnet(num_classes=low_dim)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out_logits = self.resnet(x)
        out_l2norm = self.l2norm(out_logits)
        return out_logits, out_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ProduceClass(object):

    def __init__(self, n_sample, out_dim, ratio=1.0):
        super().__init__()
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.class_per_num = self.n_sample // self.out_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = np.zeros(shape=(self.out_dim,), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample,), dtype=np.int)
        pass

    def init(self):
        class_per_num = self.n_sample // self.out_dim
        self.class_num += class_per_num
        for i in range(self.out_dim):
            self.classes[i * class_per_num: (i + 1) * class_per_num] = i
            pass
        np.random.shuffle(self.classes)
        pass

    def reset(self):
        self.count = 0
        self.count_2 = 0
        self.class_num *= 0
        pass

    def cal_label(self, out, indexes):
        top_k = out.data.topk(self.out_dim, dim=1)[1].cpu()
        indexes_cpu = indexes.cpu()

        batch_size = top_k.size(0)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int)

        for i in range(batch_size):
            for j_index, j in enumerate(top_k[i]):
                if self.class_per_num > self.class_num[j]:
                    class_labels[i] = j
                    self.class_num[j] += 1
                    self.count += 1 if self.classes[indexes_cpu[i]] != j else 0
                    self.classes[indexes_cpu[i]] = j
                    self.count_2 += 1 if j_index != 0 else 0
                    break
                pass
            pass
        pass

    def get_label(self, indexes):
        _device = indexes.device
        return torch.tensor(self.classes[indexes.cpu()]).long().to(_device)

    pass


##############################################################################################################


class RunnerTool(object):

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    @staticmethod
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif class_name.find('Linear') != -1:
            m.weight.data.normal_(0, 0.01)
            # m.bias.data = torch.ones(m.bias.data.size())
            pass
        pass

    @staticmethod
    def adjust_learning_rate1(optimizer, epoch, first_epoch, t_epoch, init_learning_rate):

        def _get_lr(_base_lr, now_epoch, _t_epoch=t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        if epoch < first_epoch + t_epoch * 0:  # 0-200
            learning_rate = init_learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 200-300
            learning_rate = init_learning_rate / 2
        elif epoch < first_epoch + t_epoch * 2:  # 300-400
            learning_rate = init_learning_rate / 4
        elif epoch < first_epoch + t_epoch * 3:  # 400-500
            learning_rate = _get_lr(init_learning_rate / 2.0, epoch - first_epoch - t_epoch * 2)
        elif epoch < first_epoch + t_epoch * 4:  # 500-600
            learning_rate = _get_lr(init_learning_rate / 4.0, epoch - first_epoch - t_epoch * 3)
        elif epoch < first_epoch + t_epoch * 5:  # 600-700
            learning_rate = _get_lr(init_learning_rate / 8.0, epoch - first_epoch - t_epoch * 4)
        elif epoch < first_epoch + t_epoch * 6:  # 700-800
            learning_rate = _get_lr(init_learning_rate / 16., epoch - first_epoch - t_epoch * 5)
        elif epoch < first_epoch + t_epoch * 7:  # 800-900
            learning_rate = _get_lr(init_learning_rate / 32., epoch - first_epoch - t_epoch * 6)
        else:  # 900-1000
            learning_rate = _get_lr(init_learning_rate / 64., epoch - first_epoch - t_epoch * 7)
            pass

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass

        return learning_rate

    @staticmethod
    def adjust_learning_rate2(optimizer, epoch, first_epoch, t_epoch, init_learning_rate):

        def _get_lr(_base_lr, now_epoch, _t_epoch=t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        if epoch < first_epoch + t_epoch * 0:  # 0-500
            learning_rate = init_learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 500-1000
            learning_rate = init_learning_rate / 10
        else:  # 1000-1500
            learning_rate = init_learning_rate / 100
            pass

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass

        return learning_rate

    pass


##############################################################################################################


class ProtoNet(nn.Module):

    def __init__(self, hid_dim, z_dim, has_norm=False, has_relu=False):
        super().__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 41
        self.conv_block_2 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 21
        self.conv_block_3 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 10
        self.conv_block_4 = nn.Sequential(nn.Conv2d(hid_dim, z_dim, 3, padding=1),
                                          nn.BatchNorm2d(z_dim), nn.ReLU(), nn.MaxPool2d(2))  # 5

        self.has_norm = has_norm
        if self.has_norm:
            self.l2norm = Normalize(2)
            self.has_relu = has_relu
            if self.has_relu:
                self.relu = nn.ReLU()
        pass

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        if self.has_norm:
            out = out.view(out.shape[0], -1)
            if self.has_relu:
                out = self.relu(out)
            out = self.l2norm(out)
        return out

    pass


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample

        self.drop_rate = drop_rate
        pass

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
            pass

        return out

    pass


class ResNet12(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        super().__init__()

        self.inplanes = 3

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate,
                                       drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate,
                                       drop_block=True, block_size=dropblock_size)

        self.keep_avg_pool = avg_pool
        self.avgpool = nn.AvgPool2d(5, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            pass

        pass

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion))
            pass

        layers = [block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size)]
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.keep_avg_pool:
            x = self.avgpool(x)

        return x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ProtoNetLarge(nn.Module):

    def __init__(self, has_norm=False):
        super().__init__()
        self.proto_net = ResNet12(block=BasicBlock, avg_pool=True, drop_rate=0.1, dropblock_size=5)

        self.has_norm = has_norm
        if self.has_norm:
            self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.proto_net(x)
        if self.has_norm:
            out = out.view(out.shape[0], -1)
            out = self.l2norm(out)
        return out

    pass


class ProtoRes18Net(nn.Module):

    def __init__(self, low_dim, has_norm=False):
        super().__init__()
        self.resnet18 = resnet18(num_classes=low_dim)

        self.has_norm = has_norm
        if self.has_norm:
            self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.resnet18(x)
        if self.has_norm:
            out = out.view(out.shape[0], -1)
            out = self.l2norm(out)
        return out

    pass


class ICProtoNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out_logits = self.linear(x)
        out_l2norm = self.l2norm(out_logits)
        return out_logits, out_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


##############################################################################################################
