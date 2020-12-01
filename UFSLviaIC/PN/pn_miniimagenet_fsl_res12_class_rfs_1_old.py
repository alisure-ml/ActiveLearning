import os
import sys
import time
import scipy
import torch
import pickle
import argparse
import platform
import warnings
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import t
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.distributions import Bernoulli
import torchvision.transforms as transforms
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pn_miniimagenet_fsl_test_tool import TestTool


##############################################################################################################


class TransformMy(object):

    @staticmethod
    def train():
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose([lambda x: Image.fromarray(x), transforms.RandomCrop(84, padding=8),
                                   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                   transforms.RandomHorizontalFlip(), lambda x: np.asarray(x),
                                   transforms.ToTensor(), normalize])

    @staticmethod
    def test():
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        return transforms.Compose([transforms.ToTensor(), normalize])

    pass


class MetaImageNet(object):

    def __init__(self, data_root, partition='train', transform=None):
        self.data_root = data_root
        self.transform = transform
        self.file_name = 'miniImageNet_category_split_{}.pickle'.format(partition)

        with open(os.path.join(self.data_root, self.file_name), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels = data['labels']
            pass

        self.n_ways = Config.n_ways
        self.n_shots = Config.n_shots
        self.n_queries = Config.n_queries
        self.n_test_runs = Config.n_test_runs

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())
        pass

    def __getitem__(self, item):
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs, support_ys, query_xs, query_ys = [], [], [], []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
            pass
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.transform(x.squeeze()), query_xs)))
        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs

    pass


##############################################################################################################


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class DropBlock(nn.Module):

    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask

    pass


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
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        pass

    def forward(self, x):
        self.num_batches_tracked += 1

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
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

    pass


class ResNet(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, num_classes=-1):
        super(ResNet, self).__init__()
        self.inplanes = 3
        self.layer1 = self._make_layer(block, n_blocks[0], 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320, stride=2,
                                       drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640, stride=2,
                                       drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block, block_size=block_size)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x
        if self.num_classes > 0:
            x = self.classifier(x)
        return feat, x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.meta_testloader = DataLoader(MetaImageNet(
            data_root=Config.data_root, partition='test', transform=TransformMy.test()),
            batch_size=1, shuffle=False, drop_last=False, num_workers=Config.num_workers)
        self.meta_valloader = DataLoader(MetaImageNet(
            data_root=Config.data_root, partition='val', transform=TransformMy.test()),
            batch_size=1, shuffle=False, drop_last=False, num_workers=Config.num_workers)

        # model
        self.model = ResNet(BasicBlock, [1, 1, 1, 1], avg_pool=True,
                            drop_rate=0.1, dropblock_size=5, num_classes=64).cuda()

        # test tool
        self.test_tool = TestTool(self.meta_proto_test, data_root=Config.data_root2, num_way=Config.n_ways,
                                  num_shot=Config.n_shots, episode_size=Config.n_queries,
                                  test_episode=Config.n_test_runs, transform=TransformMy.test())
        pass

    def validate_few_shot(self):
        start = time.time()
        val_acc = self.meta_test(self.meta_valloader)
        val_time = time.time() - start
        print('val_acc: {:.4f} time: {:.1f}'.format(val_acc, val_time))

        start = time.time()
        test_acc = self.meta_test(self.meta_testloader)
        test_time = time.time() - start
        print('test_acc: {:.4f} time: {:.1f}'.format(test_acc, test_time))
        pass

    def meta_proto_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        _, sample_z = self.model(samples)  # 5x64*5*5
        _, batch_z = self.model(batches)  # 75x64*5*5
        sample_z = self.normalize(sample_z)
        batch_z = self.normalize(batch_z)

        sample_z = sample_z.view(Config.n_ways, Config.n_shots, -1)
        batch_z = batch_z.view(batch_num, -1)
        _, z_dim = batch_z.shape

        z_proto = sample_z.mean(1)
        z_proto_expand = z_proto.unsqueeze(0).expand(batch_num, Config.n_ways, z_dim)
        z_query_expand = batch_z.unsqueeze(1).expand(batch_num, Config.n_ways, z_dim)

        dists = torch.pow(z_query_expand - z_proto_expand, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def meta_test(self, testloader):
        with torch.no_grad():
            self.model.eval()
            acc = []
            for idx, (support_xs, support_ys, query_xs, query_ys) in tqdm(enumerate(testloader)):
                support_xs, query_xs = support_xs[0].cuda(), query_xs[0].cuda()

                results = self.meta_proto_test(support_xs, query_xs)

                _, query_ys_pred = torch.max(results.data, 1)
                query_ys_pred = query_ys_pred.detach().cpu().numpy()

                acc.append(metrics.accuracy_score(query_ys.view(-1).numpy(), query_ys_pred))
            pass

        return np.mean(1.0 * np.array(acc))

    def load_model(self):
        model_path = os.path.join(Config.model_path, 'last.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            Tools.print("load model success from {}".format(model_path))
            pass
        pass

    @staticmethod
    def normalize(x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

    pass


##############################################################################################################


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore')

    batch_size = 64
    num_workers = 8

    n_test_runs = 600
    n_ways = 5
    n_shots = 1
    n_queries = 3

    # classifier = 'LR'
    classifier = 'Proto'
    # classifier = 'NN'
    # classifier = 'Cosine'
    # classifier = 'SVM'

    model_name = "{}".format(classifier)
    model_path = Tools.new_dir('../../rfs/models_pretrained/{}'.format(model_name))

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet/miniImageNet'
        data_root2 = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet/miniImageNet'
            data_root2 = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pass


"""
# Proto
epoch 100, 99.38802337646484 0.04409980621809761, total time 76.99
epoch 100, 83.74759674072266 95.8822250366211 0.6583566182879769
val_acc: 0.6540, val_std: 0.0073, time: 40.0
val_acc_feat: 0.6545, val_std: 0.0075, time: 39.7
test_acc: 0.6161, test_std: 0.0083, time: 39.0
test_acc_feat:0.5968,test_std:0.0081,time:39.3
val_acc: 0.6530, val_std: 0.0073, time: 39.0
val_acc_feat: 0.6562, val_std: 0.0076, time: 39.5
test_acc: 0.6167, test_std: 0.0082, time: 39.2
test_acc_feat:0.5944,test_std:0.0083,time:39.4

# LR
epoch 100, 99.3515625 0.045092997271567584, total time 72.31
epoch 100, 83.10752868652344 95.68487548828125 0.6670569186465481
val_acc: 0.6445, val_std: 0.0088, time: 40.3
val_acc_feat: 0.6543, val_std: 0.0085, time: 43.7
test_acc: 0.6134, test_std: 0.0088, time: 40.6
test_acc_feat:0.6163,test_std:0.0082,time:43.9
val_acc: 0.6446, val_std: 0.0089, time: 40.4
val_acc_feat: 0.6536, val_std: 0.0086, time: 43.7
test_acc: 0.6129, test_std: 0.0089, time: 40.2
test_acc_feat:0.6162,test_std:0.0081,time:43.3

# Cosine
epoch 100, 99.42448425292969 0.04334269272784392, total time 73.98
epoch 100, 83.41156768798828 95.84488677978516 0.6576445659219977
val_acc: 0.6398, val_std: 0.0094, time: 38.1
val_acc_feat: 0.6323, val_std: 0.0089, time: 38.4
test_acc: 0.6031, test_std: 0.0079, time: 38.2
test_acc_feat:0.5975,test_std:0.0076,time:38.7
val_acc: 0.6397, val_std: 0.0093, time: 38.5
val_acc_feat: 0.6315, val_std: 0.0090, time: 38.5
test_acc: 0.6038, test_std: 0.0080, time: 38.4
test_acc_feat:0.5972,test_std:0.0075,time:38.7
"""


if __name__ == '__main__':
    runner = Runner()
    # runner.train()

    runner.load_model()
    runner.model.eval()
    runner.validate_few_shot()
    runner.test_tool.val(episode=0, is_print=True)
    # runner.test_tool.test(test_avg_num=5, episode=0, is_print=True)
    runner.validate_few_shot()
    pass
