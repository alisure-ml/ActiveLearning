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
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


##############################################################################################################


class MiniImageNetIC(Dataset):

    def __init__(self, data_list, is_train=True, image_size=84):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN_PIXEL, std=Config.STD_PIXEL),
        ])
        self.transform_test = transforms.Compose([
            transforms.CenterCrop(size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN_PIXEL, std=Config.STD_PIXEL),
        ])
        self.transform = self.transform_train if is_train else self.transform_test
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = image if self.transform is None else self.transform(image)
        return image, label, idx

    pass


class MiniImageNetDataset(object):

    @staticmethod
    def get_data_all(data_root):
        train_folder = os.path.join(data_root, "train")
        val_folder = os.path.join(data_root, "val")
        test_folder = os.path.join(data_root, "test")

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

        count_image, count_class, data_val_list = 0, 0, []
        for label in os.listdir(val_folder):
            now_class_path = os.path.join(val_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_val_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        count_image, count_class, data_test_list = 0, 0, []
        for label in os.listdir(test_folder):
            now_class_path = os.path.join(test_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_test_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        return data_train_list, data_val_list, data_test_list

    pass


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


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
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


class ProduceClass(object):

    def __init__(self, n_sample, out_dim, ratio=1.0):
        super().__init__()
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.class_per_num = self.n_sample // self.out_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = np.zeros(shape=(self.out_dim, ), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample, ), dtype=np.int)
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


class KNN(object):

    @staticmethod
    def cal(labels, dist, train_labels, max_c, k, t):
        # ---------------------------------------------------------------------------------- #
        batch_size = labels.size(0)
        yd, yi = dist.topk(k + 1, dim=1, largest=True, sorted=True)
        yd, yi = yd[:, 1:], yi[:, 1:]
        retrieval = train_labels[yi]

        retrieval_1_hot = cuda(torch.zeros(k, max_c)).resize_(batch_size * k, max_c).zero_().scatter_(
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
            out_memory = cuda(torch.zeros(n_sample, low_dim).t())
            train_labels = cuda(torch.LongTensor(train_loader.dataset.train_label))
            max_c = train_labels.max() + 1

            out_list = []
            for batch_idx, (inputs, labels, indexes) in enumerate(train_loader):
                features = feature_encoder(cuda(inputs))  # 5x64*19*19
                _, out_l2norm = ic_model(features)
                out_list.append([out_l2norm, cuda(labels)])
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

            return top1 / total, top5 / total

        pass

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # data
        self.data_train, self.data_val, self.data_test = MiniImageNetDataset.get_data_all(Config.data_root)
        self.ic_train_train_loader = DataLoader(MiniImageNetIC(self.data_train, is_train=True),
                                                Config.batch_size, shuffle=True, num_workers=Config.num_workers)

        self.ic_test_train_loader = DataLoader(MiniImageNetIC(self.data_train, is_train=False),
                                               Config.batch_size, shuffle=False, num_workers=Config.num_workers)
        self.ic_test_val_loader = DataLoader(MiniImageNetIC(self.data_val, is_train=False),
                                             Config.batch_size, shuffle=False, num_workers=Config.num_workers)
        self.ic_test_test_loader = DataLoader(MiniImageNetIC(self.data_test, is_train=False),
                                              Config.batch_size, shuffle=False, num_workers=Config.num_workers)

        # model
        self.feature_encoder = cuda(CNNEncoder())
        self.ic_model = cuda(ICModel(in_dim=Config.ic_in_dim, out_dim=Config.ic_out_dim))
        self.ic_loss = cuda(nn.CrossEntropyLoss())
        # self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=Config.learning_rate)
        # self.ic_model_optim = torch.optim.Adam(self.ic_model.parameters(), lr=Config.learning_rate)
        self.feature_encoder_optim = torch.optim.SGD(self.feature_encoder.parameters(),
                                                     lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.ic_model_optim = torch.optim.SGD(self.ic_model.parameters(),
                                              lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()
        pass

    @staticmethod
    def _adjust_learning_rate(optimizer, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=Config.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = Config.t_epoch
        first_epoch = Config.first_epoch
        init_learning_rate = Config.learning_rate

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

    @staticmethod
    def _adjust_learning_rate2(optimizer, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=Config.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = Config.t_epoch
        first_epoch = Config.first_epoch
        init_learning_rate = Config.learning_rate
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

    def load_model(self):
        if os.path.exists(Config.fe_dir):
            self.feature_encoder.load_state_dict(torch.load(Config.fe_dir))
            Tools.print("load feature encoder success from {}".format(Config.fe_dir))

        if os.path.exists(Config.ic_dir):
            self.ic_model.load_state_dict(torch.load(Config.ic_dir))
            Tools.print("load ic model success from {}".format(Config.ic_dir))
        pass

    def train(self):
        Tools.print()
        Tools.print("Training...")

        # Init Update
        try:
            self.feature_encoder.eval()
            self.ic_model.eval()
            Tools.print("Init label {} .......")
            self.produce_class.reset()
            for image, label, idx in tqdm(self.ic_train_loader):
                image, label, idx = cuda(image), cuda(label), cuda(idx)
                features = self.feature_encoder(image)  # 5x64*19*19
                ic_out_logits, ic_out_l2norm = self.ic_model(features)
                self.produce_class.cal_label(ic_out_l2norm, idx)
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2))
        finally:
            pass

        for epoch in range(Config.train_epoch):
            self.feature_encoder.train()
            self.ic_model.train()

            Tools.print()
            fe_lr= self._adjust_learning_rate(self.feature_encoder_optim, epoch)
            ic_lr = self._adjust_learning_rate(self.ic_model_optim, epoch)
            Tools.print('Epoch: [{}] fe_lr={} ic_lr={}'.format(epoch, fe_lr, ic_lr))

            all_loss = 0.0
            self.produce_class.reset()
            for image, label, idx in tqdm(self.ic_train_train_loader):
                image, label, idx = cuda(image), cuda(label), cuda(idx)

                ###########################################################################
                # 1 calculate features
                features = self.feature_encoder(image)  # 5x64*19*19
                ic_out_logits, ic_out_l2norm = self.ic_model(features)

                ic_targets = self.produce_class.get_label(idx)
                self.produce_class.cal_label(ic_out_l2norm, idx)

                # 2 loss
                loss = self.ic_loss(ic_out_logits, ic_targets)
                all_loss += loss.item()

                # 3 backward
                self.feature_encoder.zero_grad()
                self.ic_model.zero_grad()
                loss.backward()
                self.feature_encoder_optim.step()
                self.ic_model_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f}".format(epoch + 1, all_loss / len(self.ic_train_train_loader)))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                self.feature_encoder.eval()
                self.ic_model.eval()
                Tools.print()
                Tools.print("Test {} .......".format(epoch))
                val_accuracy = self.val_ic(epoch, ic_loader=self.ic_test_train_loader, name="Train")
                self.val_ic(epoch, ic_loader=self.ic_test_val_loader, name="Val")
                self.val_ic(epoch, ic_loader=self.ic_test_test_loader, name="Test")
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.feature_encoder.state_dict(), Config.fe_dir)
                    torch.save(self.ic_model.state_dict(), Config.ic_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    def val_ic(self, epoch, ic_loader, name="Test"):
        acc_1, acc_2 = KNN.knn(self.feature_encoder, self.ic_model, Config.ic_out_dim, ic_loader, 100)
        Tools.print("Epoch: [{}] {} {:.4f}/{:.4f}".format(epoch, name, acc_1, acc_2))
        return acc_1

    pass


##############################################################################################################


"""
1_64_512_2_0.001
2020-10-22 01:38:55 Train: 23038/2081
2020-10-22 01:39:03 Epoch: [1000] Train 0.3233/0.6380
2020-10-22 01:39:06 Epoch: [1000]   Val 0.4824/0.8655
2020-10-22 01:39:08 Epoch: [1000]  Test 0.4632/0.8455

1_64_512_2_200_100_0.01_fe
2020-10-22 17:11:26 Train: [880] 22872/2055
2020-10-22 17:11:37 Epoch: [880] Train 0.3388/0.6592
2020-10-22 17:11:41 Epoch: [880]   Val 0.4918/0.8751
2020-10-22 17:11:45 Epoch: [880]  Test 0.4650/0.8552

1_64_512_2_500_500_0.01_fe
2020-10-22 22:14:17 Train: [1450] 22774/2081
2020-10-22 22:14:24 Epoch: [1450] Train 0.3422/0.6623
2020-10-22 22:14:26 Epoch: [1450] Val 0.4956/0.8808
2020-10-22 22:14:29 Epoch: [1450] Test 0.4669/0.8520
"""


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    num_workers = 8
    batch_size = 64
    val_freq = 10

    learning_rate = 0.01

    # train_epoch = 1000
    # first_epoch, t_epoch = 200, 100

    train_epoch = 1500
    first_epoch, t_epoch = 500, 500

    # ic
    ic_in_dim = 64
    ic_out_dim = 512
    ic_ratio = 2

    model_name = "1_{}_{}_{}_{}_{}_{}".format(batch_size, ic_out_dim, ic_ratio, first_epoch, t_epoch, learning_rate)

    MEAN_PIXEL = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    STD_PIXEL = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    fe_dir = Tools.new_dir("../models/ic/{}_fe.pkl".format(model_name))
    ic_dir = Tools.new_dir("../models/ic/{}_ic.pkl".format(model_name))
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.feature_encoder.eval()
    runner.ic_model.eval()
    runner.val_ic(0, ic_loader=runner.ic_test_train_loader, name="Train")
    runner.val_ic(0, ic_loader=runner.ic_test_val_loader, name="Val")
    runner.val_ic(0, ic_loader=runner.ic_test_test_loader, name="Test")

    runner.train()

    runner.load_model()
    runner.feature_encoder.eval()
    runner.ic_model.eval()
    runner.val_ic(epoch=Config.train_epoch, ic_loader=runner.ic_test_train_loader, name="Final Train")
    runner.val_ic(epoch=Config.train_epoch, ic_loader=runner.ic_test_val_loader, name="Final Val")
    runner.val_ic(epoch=Config.train_epoch, ic_loader=runner.ic_test_test_loader, name="Final Test")
    pass
