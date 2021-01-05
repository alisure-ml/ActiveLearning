import os
import math
import torch
import random
import platform
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from collections import Counter
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader


##############################################################################################################
# Train Dataset


class MyDataset(object):

    dataset_name_miniimagenet = "miniimagenet"
    dataset_name_tieredimagenet = "tieredimagenet"

    dataset_split_train = "train"
    dataset_split_val = "val"
    dataset_split_test = "test"

    @staticmethod
    def get_data_split(data_root, split="train"):
        train_folder = os.path.join(data_root, split)

        count_image, count_class, data_train_list = 0, 0, []
        for label in os.listdir(train_folder):
            now_class_path = os.path.join(train_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_train_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        return data_train_list

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    @staticmethod
    def get_data_root(dataset_name, is_png=True):
        if dataset_name == MyDataset.dataset_name_miniimagenet:
            if "Linux" in platform.platform():
                data_root = '/mnt/4T/Data/data/miniImagenet'
                if not os.path.isdir(data_root):
                    data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
            else:
                data_root = "F:\\data\\miniImagenet"
            data_root = os.path.join(data_root, "miniImageNet_png") if is_png else data_root
        elif dataset_name == MyDataset.dataset_name_tieredimagenet:
            if "Linux" in platform.platform():
                data_root = '/mnt/4T/Data/data/UFSL/tiered-imagenet'
                if not os.path.isdir(data_root):
                    data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/tiered-imagenet'
            else:
                data_root = "F:\\data\\UFSL\\tiered-imagenet"
        else:
            raise Exception("..........................")
        return data_root

    @staticmethod
    def get_ways_shots(dataset_name, split):
        if dataset_name == MyDataset.dataset_name_miniimagenet:
            if split == MyDataset.dataset_split_test:
                ways = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                shots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            elif split == MyDataset.dataset_split_val:
                ways = [2, 5, 10, 15, 16]
                shots = [1, 5, 10, 15, 20, 30, 40, 50]
            else:
                ways = [2, 5, 10, 15, 20, 30, 50]
                shots = [1, 5, 10, 15, 20, 30, 40, 50]
        elif dataset_name == MyDataset.dataset_name_tieredimagenet:
            if split == MyDataset.dataset_split_test:
                ways = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 35, 30, 35, 40, 45, 50]
                shots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            elif split == MyDataset.dataset_split_val:
                ways = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 35, 30, 35, 40, 45, 50]
                shots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            else:
                ways = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 35, 30, 35, 40, 45, 50]
                shots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        else:
            raise Exception(".")
        return ways, shots
    pass


class MyTransforms(object):

    @staticmethod
    def get_transform_miniimagenet(normalize, has_ic=True, is_fsl_simple=True, is_css=False):
        transform_train_ic = transforms.Compose([
            # transforms.RandomCrop(84, padding=8),
            transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

        transform_train_fsl_simple = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_train_fsl_hard = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_train_fsl_css = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(84),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_train_fsl = transform_train_fsl_simple if is_fsl_simple else transform_train_fsl_hard
        transform_train_fsl = transform_train_fsl_css if is_css else transform_train_fsl

        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if has_ic:
            return transform_train_ic, transform_train_fsl, transform_test
        else:
            return transform_train_fsl, transform_test
        pass

    @classmethod
    def get_transform_tieredimagenet(cls, normalize, has_ic=True, is_fsl_simple=True, is_css=False):
        return cls.get_transform_miniimagenet(normalize, has_ic, is_fsl_simple, is_css)

    @classmethod
    def get_transform(cls, dataset_name, has_ic=True, is_fsl_simple=True, is_css=False):
        normalize_1 = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                             std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        normalize_2 = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        if dataset_name == MyDataset.dataset_name_miniimagenet:
            return cls.get_transform_miniimagenet(normalize_1, has_ic=has_ic,
                                                  is_fsl_simple=is_fsl_simple, is_css=is_css)
        elif dataset_name == MyDataset.dataset_name_tieredimagenet:
            return cls.get_transform_tieredimagenet(normalize_1, has_ic=has_ic,
                                                    is_fsl_simple=is_fsl_simple, is_css=is_css)
        else:
            raise Exception("......")
        pass

    pass


class TrainDataset(object):

    def __init__(self, data_list, num_way, num_shot, transform_train_ic, transform_train_fsl, transform_test):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot
        self.data_id = np.asarray(range(len(self.data_list)))

        self.classes = None

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        self.transform_train_ic = transform_train_ic
        self.transform_train_fsl = transform_train_fsl
        self.transform_test = transform_test
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, _, now_image_filename = now_label_image_tuple
        _now_label = self.classes[item]

        now_label_k_shot_index = self._get_samples_by_clustering_label(_now_label, True, num=self.num_shot)
        is_ok_list = [self.data_list[one][1] == now_label_image_tuple[1] for one in now_label_k_shot_index]

        # 其他样本
        other_label_k_shot_index_list = self._get_samples_by_clustering_label(_now_label, False,
                                                                              num=self.num_shot * (self.num_way - 1))

        # c_way_k_shot
        c_way_k_shot_index_list = now_label_k_shot_index + other_label_k_shot_index_list
        random.shuffle(c_way_k_shot_index_list)

        if len(c_way_k_shot_index_list) != self.num_shot * self.num_way:
            return self.__getitem__(random.sample(list(range(0, len(self.data_list))), 1)[0])

        task_list = [self.data_list[index] for index in c_way_k_shot_index_list] + [now_label_image_tuple]

        task_data = []
        for one in task_list:
            transform = self.transform_train_ic if one[2] == now_image_filename else self.transform_train_fsl
            task_data.append(torch.unsqueeze(MyDataset.read_image(one[2], transform), dim=0))
            pass
        task_data = torch.cat(task_data)

        task_label = torch.Tensor([int(index in now_label_k_shot_index) for index in c_way_k_shot_index_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index, is_ok_list

    def _get_samples_by_clustering_label(self, label, is_same_label=False, num=1):
        if is_same_label:
            return random.sample(list(np.squeeze(np.argwhere(self.classes == label), axis=1)), num)
        else:
            return random.sample(list(np.squeeze(np.argwhere(self.classes != label))), num)
        pass

    def set_samples_class(self, classes):
        self.classes = classes
        pass

    pass


##############################################################################################################
# IC


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
# Net


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


class C4Net(nn.Module):

    def __init__(self, hid_dim, z_dim, has_norm=False):
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
            self.norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)

        if self.has_norm:
            out = out.view(out.shape[0], -1)
            out = self.norm(out)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0):
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


class ResNet12Small(nn.Module):

    def __init__(self, block=BasicBlock, avg_pool=True, drop_rate=0.1):
        super().__init__()
        self.inplanes = 3

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, stride=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, stride=2, drop_rate=drop_rate)

        self.keep_avg_pool = avg_pool
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            pass

        pass

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion))
            pass

        layers = [block(self.inplanes, planes, stride, downsample, drop_rate)]
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) if self.keep_avg_pool else x
        return x

    pass


class ICResNet(nn.Module):

    def __init__(self, resnet, low_dim=512, modify_head=False):
        super().__init__()
        self.resnet = resnet(num_classes=low_dim)
        self.l2norm = Normalize(2)
        if modify_head:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            pass
        pass

    def forward(self, x):
        out_logits = self.resnet(x)
        out_l2norm = self.l2norm(out_logits)
        return out_logits, out_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


##############################################################################################################
# Runner


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
# Test


class ClassBalancedSampler(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
            pass

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
            pass

        return iter(batch)

    def __len__(self):
        return 1

    pass


class ClassBalancedSamplerTest(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_cl, num_inst, shuffle=True):
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i + j * self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
            pass

        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
                pass
            pass

        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1

    pass


class Task(object):
    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = dict(zip(class_folders, np.array(range(len(class_folders)))))

        samples = dict()
        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num + test_num]
            pass

        self.train_labels = [labels[os.path.split(x)[0]] for x in self.train_roots]
        self.test_labels = [labels[os.path.split(x)[0]] for x in self.test_roots]
        pass

    pass


class TestDataset(Dataset):
    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image = Image.open(self.image_roots[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    @staticmethod
    def folders(data_root):
        train_folder = os.path.join(data_root, MyDataset.dataset_split_train)
        val_folder = os.path.join(data_root, MyDataset.dataset_split_val)
        test_folder = os.path.join(data_root, MyDataset.dataset_split_test)

        folders_train = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]
        folders_val = [os.path.join(val_folder, label) for label in os.listdir(val_folder)
                       if os.path.isdir(os.path.join(val_folder, label))]
        folders_test = [os.path.join(test_folder, label) for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))]

        random.seed(1)
        random.shuffle(folders_train)
        random.shuffle(folders_val)
        random.shuffle(folders_test)
        return folders_train, folders_val, folders_test

    @staticmethod
    def get_data_loader(task, num_per_class=1, split="train", sampler_test=False, shuffle=False, transform=None):
        if split == "train":
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        if transform is None:
            raise Exception("Note that transform is None")

        dataset = TestDataset(task, split=split, transform=transform)
        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    pass


class FSLTestTool(object):

    def __init__(self, model_fn, data_root, num_way=5, num_shot=1,
                 episode_size=15, test_episode=600, transform=None, txt_path=None):
        self.model_fn = model_fn
        self.transform = transform
        self.txt_path = txt_path

        self.folders_train, self.folders_val, self.folders_test = TestDataset.folders(data_root)

        self.test_episode = test_episode
        self.num_way = num_way
        self.num_shot = num_shot
        self.episode_size = episode_size
        pass

    @staticmethod
    def _compute_confidence_interval(data):
        a = 1.0 * np.array(data)
        m = np.mean(a)
        std = np.std(a)
        pm = 1.96 * (std / np.sqrt(len(a)))
        return m, pm

    def eval(self, num_way=5, num_shot=1, episode_size=15, test_episode=1000):
        acc_list = self._val_no_mean(self.folders_test, sampler_test=True, num_way=num_way,
                                     num_shot=num_shot, episode_size=episode_size, test_episode=test_episode)
        m, pm = self._compute_confidence_interval(acc_list)
        return m, pm

    def val_train(self):
        return self._val(self.folders_train, sampler_test=False, all_episode=self.test_episode)

    def val_val(self):
        return self._val(self.folders_val, sampler_test=False, all_episode=self.test_episode)

    def val_test(self):
        return self._val(self.folders_test, sampler_test=False, all_episode=self.test_episode)

    def val_test2(self):
        return self._val(self.folders_test, sampler_test=True, all_episode=self.test_episode)

    def test(self, test_avg_num, episode=0, is_print=True):
        acc_list = []
        for _ in range(test_avg_num):
            acc = self._val(self.folders_test, sampler_test=True, all_episode=self.test_episode)
            acc_list.append(acc)
            pass

        mean_acc = np.mean(acc_list)
        if is_print:
            for acc in acc_list:
                Tools.print("episode={}, Test accuracy={}".format(episode, acc), txt_path=self.txt_path)
                pass
            Tools.print("episode={}, Mean Test accuracy={}".format(episode, mean_acc), txt_path=self.txt_path)
            pass
        return mean_acc

    def val(self, episode=0, is_print=True, has_test=True):
        acc_train = self.val_train()
        if is_print:
            Tools.print("Train {} Accuracy: {}".format(episode, acc_train), txt_path=self.txt_path)

        acc_val = self.val_val()
        if is_print:
            Tools.print("Val   {} Accuracy: {}".format(episode, acc_val), txt_path=self.txt_path)

        acc_test1 = self.val_test()
        if is_print:
            Tools.print("Test1 {} Accuracy: {}".format(episode, acc_test1), txt_path=self.txt_path)

        if has_test:
            acc_test2 = self.val_test2()
            if is_print:
                Tools.print("Test2 {} Accuracy: {}".format(episode, acc_test2), txt_path=self.txt_path)
                pass
        return acc_val

    def _val(self, folders, sampler_test, all_episode):
        accuracies = self._val_no_mean(folders, sampler_test, num_way=self.num_way, num_shot=self.num_shot,
                                       episode_size=self.episode_size, test_episode=all_episode)
        return np.mean(np.array(accuracies, dtype=np.float))

    def _val_no_mean(self, folders, sampler_test, num_way, num_shot, episode_size, test_episode):
        accuracies = []
        for i in range(test_episode):
            total_rewards = 0
            counter = 0
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本
            task = Task(folders, num_way, num_shot, episode_size)
            sample_data_loader = TestDataset.get_data_loader(task, num_shot, "train", sampler_test=sampler_test,
                                                           shuffle=False, transform=self.transform)
            num_per_class = 5 if num_shot > 1 else 3
            batch_data_loader = TestDataset.get_data_loader(task, num_per_class, "val", sampler_test=sampler_test,
                                                          shuffle=True, transform=self.transform)
            samples, labels = sample_data_loader.__iter__().next()

            with torch.no_grad():
                samples = RunnerTool.to_cuda(samples)
                for batches, batch_labels in batch_data_loader:
                    results = self.model_fn(samples, RunnerTool.to_cuda(batches), num_way=num_way, num_shot=num_shot)

                    _, predict_labels = torch.max(results.data, 1)
                    batch_size = batch_labels.shape[0]
                    rewards = [1 if predict_labels[j].cpu() == batch_labels[j] else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)

                    counter += batch_size
                    pass
                pass

            accuracies.append(total_rewards / 1.0 / counter)
            pass
        return accuracies

    pass


class ICDataset(Dataset):

    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]
        self.transform = transform
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = image if self.transform is None else self.transform(image)
        return image, label, idx

    pass


class KNN(object):

    @classmethod
    def cal(cls, labels, dist, train_labels, max_c, k, t):
        # ---------------------------------------------------------------------------------- #
        batch_size = labels.size(0)
        yd, yi = dist.topk(k + 1, dim=1, largest=True, sorted=True)
        yd, yi = yd[:, 1:], yi[:, 1:]
        retrieval = train_labels[yi]

        retrieval_1_hot = RunnerTool.to_cuda(torch.zeros(k, max_c)).resize_(batch_size * k, max_c).zero_().scatter_(
            1, retrieval.view(-1, 1), 1).view(batch_size, -1, max_c)
        yd_transform = yd.clone().div_(t).exp_().view(batch_size, -1, 1)
        probs = torch.sum(torch.mul(retrieval_1_hot, yd_transform), 1)
        _, predictions = probs.sort(1, True)
        # ---------------------------------------------------------------------------------- #

        correct = predictions.eq(labels.data.view(-1, 1))

        top1 = correct.narrow(1, 0, 1).sum().item()
        top5 = correct.narrow(1, 0, 5).sum().item()
        return top1, top5

    @classmethod
    def knn(cls, feature_encoder, ic_model, low_dim, train_loader, k, t=0.1):

        with torch.no_grad():
            n_sample = train_loader.dataset.__len__()
            out_memory = RunnerTool.to_cuda(torch.zeros(n_sample, low_dim).t())
            train_labels = RunnerTool.to_cuda(torch.LongTensor(train_loader.dataset.train_label))
            max_c = train_labels.max() + 1

            # clustering 1
            clustering = np.zeros(n_sample, dtype=np.int)

            out_list = []
            for batch_idx, (inputs, labels, indexes) in enumerate(train_loader):
                inputs = RunnerTool.to_cuda(inputs)

                if feature_encoder is None:
                    _, out_l2norm = ic_model(inputs)
                else:
                    features = feature_encoder(inputs)  # 5x64*19*19
                    _, out_l2norm = ic_model(features)
                    pass

                # clustering 2
                now_clustering = torch.argmax(out_l2norm, dim=1).cpu()
                clustering[indexes] = now_clustering

                out_list.append([out_l2norm, RunnerTool.to_cuda(labels)])
                out_memory[:, batch_idx * inputs.size(0):(batch_idx + 1) * inputs.size(0)] = out_l2norm.data.t()
                pass

            top1, top5, total = 0., 0., 0
            for out in out_list:
                dist = torch.mm(out[0], out_memory)
                _top1, _top5 = cls.cal(out[1], dist, train_labels, max_c, k, t)

                top1 += _top1
                top5 += _top5
                total += out[1].size(0)
                pass

            # clustering 3
            acc_cluster = cls.cluster_acc(clustering, train_labels.cpu().long())

            return top1 / total, top5 / total, acc_cluster

        pass

    @staticmethod
    def cluster_acc(clustering, train_labels):
        counter_dict = {}
        for index, value in enumerate(clustering):
            if value not in counter_dict:
                counter_dict[value] = []
            counter_dict[value].append(int(train_labels[index]))
            pass
        for key in counter_dict:
            counter_dict[key] = dict(Counter(counter_dict[key]))
            pass
        return 0

    pass


class ICTestTool(object):

    def __init__(self, feature_encoder, ic_model, data_root, transform,
                 batch_size=64, num_workers=8, ic_out_dim=512, k=100, txt_path=None):
        self.feature_encoder = feature_encoder if feature_encoder is None else RunnerTool.to_cuda(feature_encoder)
        self.ic_model = RunnerTool.to_cuda(ic_model)
        self.ic_out_dim = ic_out_dim
        self.k = k
        self.txt_path = txt_path

        self.data_train = MyDataset.get_data_split(data_root, split=MyDataset.dataset_split_train)
        self.train_loader = DataLoader(ICDataset(self.data_train, transform), batch_size, False, num_workers=num_workers)

        self.data_val = MyDataset.get_data_split(data_root, split=MyDataset.dataset_split_val)
        self.val_loader = DataLoader(ICDataset(self.data_val, transform), batch_size, False, num_workers=num_workers)

        self.data_test = MyDataset.get_data_split(data_root, split=MyDataset.dataset_split_test)
        self.test_loader = DataLoader(ICDataset(self.data_test, transform), batch_size, False, num_workers=num_workers)
        pass

    def val_ic(self, ic_loader):
        acc_1, acc_2, acc_3 = KNN.knn(self.feature_encoder, self.ic_model, self.ic_out_dim, ic_loader, self.k)
        return acc_1, acc_2, acc_3

    def val(self, epoch, is_print=True):
        if is_print:
            Tools.print()
            Tools.print("Test {} .......".format(epoch), txt_path=self.txt_path)
            pass

        acc_1_train, acc_2_train, acc_3_train = self.val_ic(ic_loader=self.train_loader)
        if is_print:
            Tools.print("Epoch: {} Train {:.4f}/{:.4f} {:.4f}".format(epoch, acc_1_train, acc_2_train, acc_3_train), txt_path=self.txt_path)

        acc_1_val, acc_2_val, acc_3_val = self.val_ic(ic_loader=self.val_loader)
        if is_print:
            Tools.print("Epoch: {} Val   {:.4f}/{:.4f} {:.4f}".format(epoch, acc_1_val, acc_2_val, acc_3_val), txt_path=self.txt_path)

        acc_1_test, acc_2_test, acc_3_test = self.val_ic(ic_loader=self.test_loader)
        if is_print:
            Tools.print("Epoch: {} Test  {:.4f}/{:.4f} {:.4f}".format(epoch, acc_1_test, acc_2_test, acc_3_test), txt_path=self.txt_path)
            pass
        return acc_1_val

    pass


class EvalDataset(Dataset):

    def __init__(self, task, image_features, split='train'):
        self.task = task
        self.split = split
        self.image_features = image_features

        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image = self.image_features[self.image_roots[idx]]
        label = self.labels[idx]
        return image, label

    @staticmethod
    def folders(data_root):
        train_folder = os.path.join(data_root, MyDataset.dataset_split_train)
        val_folder = os.path.join(data_root, MyDataset.dataset_split_val)
        test_folder = os.path.join(data_root, MyDataset.dataset_split_test)

        folders_train = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]
        folders_val = [os.path.join(val_folder, label) for label in os.listdir(val_folder)
                       if os.path.isdir(os.path.join(val_folder, label))]
        folders_test = [os.path.join(test_folder, label) for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))]

        random.seed(1)
        random.shuffle(folders_train)
        random.shuffle(folders_val)
        random.shuffle(folders_test)
        return folders_train, folders_val, folders_test

    @staticmethod
    def get_data_loader(task, image_features, num_per_class=1, split="train", sampler_test=False, shuffle=False):
        if split == "train":
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        dataset = EvalDataset(task, image_features=image_features, split=split)
        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    pass


class FSLEvalTool(object):

    def __init__(self, model_fn, data_root, image_features, num_way=5, num_shot=1,
                 episode_size=15, test_episode=600, txt_path=None):
        self.model_fn = model_fn
        self.image_features = image_features
        self.txt_path = txt_path

        self.folders_train, self.folders_val, self.folders_test = TestDataset.folders(data_root)

        self.test_episode = test_episode
        self.num_way = num_way
        self.num_shot = num_shot
        self.episode_size = episode_size
        pass

    @staticmethod
    def _compute_confidence_interval(data):
        a = 1.0 * np.array(data)
        m = np.mean(a)
        std = np.std(a)
        pm = 1.96 * (std / np.sqrt(len(a)))
        return m, pm

    def eval(self, num_way=5, num_shot=1, episode_size=15, test_episode=1000, split=MyDataset.dataset_split_test):
        folders_test = self.folders_test
        if split == MyDataset.dataset_split_train:
            folders_test = self.folders_train
        elif split == MyDataset.dataset_split_val:
            folders_test = self.folders_val
            pass

        acc_list = self._val_no_mean(folders_test, sampler_test=True, num_way=num_way,
                                     num_shot=num_shot, episode_size=episode_size, test_episode=test_episode)
        m, pm = self._compute_confidence_interval(acc_list)
        return m, pm

    def _val_no_mean(self, folders, sampler_test, num_way, num_shot, episode_size, test_episode):
        accuracies = []
        for i in range(test_episode):
            total_rewards = 0
            counter = 0
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本
            task = Task(folders, num_way, num_shot, episode_size)
            sample_data_loader = EvalDataset.get_data_loader(task, self.image_features, num_shot, "train",
                                                             sampler_test=sampler_test, shuffle=False)
            num_per_class = 5 if num_shot > 1 else 3
            batch_data_loader = EvalDataset.get_data_loader(task, self.image_features, num_per_class, "val",
                                                            sampler_test=sampler_test, shuffle=True)
            samples, labels = sample_data_loader.__iter__().next()

            with torch.no_grad():
                samples = RunnerTool.to_cuda(samples)
                for batches, batch_labels in batch_data_loader:
                    results = self.model_fn(samples, RunnerTool.to_cuda(batches), num_way=num_way, num_shot=num_shot)

                    _, predict_labels = torch.max(results.data, 1)
                    batch_size = batch_labels.shape[0]
                    rewards = [1 if predict_labels[j].cpu() == batch_labels[j] else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)

                    counter += batch_size
                    pass
                pass

            accuracies.append(total_rewards / 1.0 / counter)
            pass
        return accuracies

    pass


class EvalFeatureDataset(Dataset):

    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_filename = self.data_list[idx][-1]
        image = Image.open(image_filename).convert('RGB')
        image = self.transform(image)
        return image, image_filename

    pass


##############################################################################################################

