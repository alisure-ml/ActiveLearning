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


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-16)
        return out

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
