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
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


##############################################################################################################


class MiniImageNetIC(Dataset):

    def __init__(self, data_list, is_train=True):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN_PIXEL, std=Config.STD_PIXEL),
        ])
        self.transform_test = transforms.Compose([
            transforms.CenterCrop(size=84),
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


class ICResNet(nn.Module):

    def __init__(self, low_dim=512):
        super().__init__()
        self.resnet = resnet18(num_classes=low_dim)
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
    def knn(cls, ic_model, low_dim, train_loader, k, t=0.1):

        with torch.no_grad():
            n_sample = train_loader.dataset.__len__()
            out_memory = cuda(torch.zeros(n_sample, low_dim).t())
            train_labels = cuda(torch.LongTensor(train_loader.dataset.train_label))
            max_c = train_labels.max() + 1

            out_list = []
            for batch_idx, (inputs, labels, indexes) in enumerate(train_loader):
                _, out_l2norm = ic_model(cuda(inputs))
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
        self.ic_model = cuda(ICResNet(Config.ic_out_dim))
        self.ic_loss = cuda(nn.CrossEntropyLoss())
        self.ic_model_optim = torch.optim.Adam(self.ic_model.parameters(), lr=Config.learning_rate)

        # IC
        self.produce_class1 = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class2 = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class1.init()
        self.produce_class2.init()
        pass

    def load_model(self):
        if os.path.exists(Config.ic_dir):
            self.ic_model.load_state_dict(torch.load(Config.ic_dir))
            Tools.print("load ic model success from {}".format(Config.ic_dir))
        pass

    def train(self):
        Tools.print()
        Tools.print("Training...")

        for epoch in range(Config.train_epoch):
            self.ic_model.train()

            Tools.print()
            self.produce_class1.reset()
            all_loss = 0.0
            for image, label, idx in tqdm(self.ic_train_train_loader):
                image, label, idx = cuda(image), cuda(label), cuda(idx)

                ###########################################################################
                # 1 calculate features
                ic_out_logits, ic_out_l2norm = self.ic_model(image)

                # 2 calculate labels
                self.produce_class1.cal_label(ic_out_l2norm, idx)
                ic_targets = self.produce_class2.get_label(idx)

                # 3 loss
                loss = self.ic_loss(ic_out_logits, ic_targets)
                all_loss += loss.item()

                # 4 backward
                self.ic_model.zero_grad()
                loss.backward()
                self.ic_model_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} lr:{}".format(
                epoch + 1, all_loss / len(self.ic_train_train_loader), Config.learning_rate))
            ###########################################################################

            ###########################################################################
            # 切换
            classes = self.produce_class2.classes
            self.produce_class2.classes = self.produce_class1.classes
            self.produce_class1.classes = classes
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class1.count, self.produce_class1.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                self.ic_model.eval()
                Tools.print()
                Tools.print("Test {} .......".format(epoch))
                val_accuracy = self.val_ic(epoch, ic_loader=self.ic_test_train_loader, name="Train")
                self.val_ic(epoch, ic_loader=self.ic_test_val_loader, name="Val")
                self.val_ic(epoch, ic_loader=self.ic_test_test_loader, name="Test")
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.ic_model.state_dict(), Config.ic_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    def val_ic(self, epoch, ic_loader, name="Test"):
        acc_1, acc_2 = KNN.knn(self.ic_model, Config.ic_out_dim, ic_loader, 100)
        Tools.print("Epoch: [{}] {} {:.4f}/{:.4f}".format(epoch, name, acc_1, acc_2))
        return acc_1

    pass


##############################################################################################################
"""
2_64_200_1_0.001
2020-10-20 21:02:27 Test 237 .......
2020-10-20 21:02:42 Epoch: [237] Train 0.3524/0.6671
2020-10-20 21:02:46 Epoch: [237] Val 0.4882/0.8697
2020-10-20 21:02:51 Epoch: [237] Test 0.4667/0.8555
2020-10-20 21:02:51 Save networks for epoch: 237

2_64_128_2_0.001
2020-10-21 08:12:53 Test 830 .......
2020-10-21 08:13:09 Epoch: [830] Train 0.3792/0.6978
2020-10-21 08:13:13 Epoch: [830] Val 0.5069/0.8831
2020-10-21 08:13:18 Epoch: [830] Test 0.4816/0.8652
2020-10-21 08:13:18 Save networks for epoch: 830
2020-10-21 10:19:59 load ic model success from ../models/ic/2_64_128_2_0.001_ic.pkl
2020-10-21 10:20:15 Epoch: [1000] Final Train 0.3792/0.6978
2020-10-21 10:20:19 Epoch: [1000] Final Val 0.5069/0.8831
2020-10-21 10:20:24 Epoch: [1000] Final Test 0.4816/0.8652
"""


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    train_epoch = 1000
    learning_rate = 0.001
    num_workers = 8

    batch_size = 64

    val_freq = 10

    # ic
    ic_out_dim = 128
    ic_ratio = 2

    model_name = "2_{}_{}_{}_{}".format(batch_size, ic_out_dim, ic_ratio, learning_rate)

    MEAN_PIXEL = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    STD_PIXEL = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    ic_dir = Tools.new_dir("../models/ic/{}_ic.pkl".format(model_name))
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.ic_model.eval()
    runner.val_ic(0, ic_loader=runneric_test_train_loader, name="Train")
    runner.val_ic(0, ic_loader=runner.ic_test_val_loader, name="Val")
    runner.val_ic(0, ic_loader=runner.ic_test_test_loader, name="Test")

    runner.train()

    runner.load_model()
    runner.ic_model.eval()
    runner.val_ic(epoch=Config.train_epoch, ic_loader=runner.ic_test_train_loader, name="Final Train")
    runner.val_ic(epoch=Config.train_epoch, ic_loader=runner.ic_test_val_loader, name="Final Val")
    runner.val_ic(epoch=Config.train_epoch, ic_loader=runner.ic_test_test_loader, name="Final Test")
    pass
