import os
import sys
import math
import torch
import random
import platform
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from tool_ic_test import ICTestTool
from alisuretool.Tools import Tools
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
sys.path.append("../Common")
from UFSLTool import C4Net, RunnerTool, Normalize


##############################################################################################################


class DatasetIC(Dataset):

    def __init__(self, data_list, image_size):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
        self.transform = transforms.Compose([transforms.RandomRotation(30, fill=255),
                                             transforms.Resize(image_size),
                                             transforms.RandomCrop(image_size, padding=4, fill=255),
                                             transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = self.transform(image)
        return image, label, idx

    @staticmethod
    def get_data_all(data_root):
        train_folder = os.path.join(data_root, "train")

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

    pass


##############################################################################################################


class EncoderC4(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = C4Net(64, 64)
        self.out_dim = 64
        pass

    def forward(self, x):
        out = self.encoder(x)
        out = torch.flatten(out, 1)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ICResNet(nn.Module):

    def __init__(self, encoder, low_dim=512):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(self.encoder.out_dim, low_dim)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.encoder(x)
        out_logits = self.fc(out)
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


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0
        self.adjust_learning_rate = Config.adjust_learning_rate

        # data
        self.data_train = DatasetIC.get_data_all(Config.data_root)
        self.ic_train_loader = DataLoader(DatasetIC(self.data_train, image_size=Config.image_size),
                                          Config.batch_size, shuffle=True, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()

        # model
        self.ic_model = self.to_cuda(ICResNet(low_dim=Config.ic_out_dim, encoder=Config.ic_net))
        self.ic_loss = self.to_cuda(nn.CrossEntropyLoss())

        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # Eval
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
                                       data_root=Config.data_root, batch_size=Config.batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim,
                                       transform=self.ic_train_loader.dataset.transform_test, k=Config.knn)
        pass

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    @staticmethod
    def _weights_init(m):
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
    def adjust_learning_rate1(optimizer, epoch):

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

    @staticmethod
    def adjust_learning_rate2(optimizer, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=Config.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = Config.t_epoch
        first_epoch = Config.first_epoch
        init_learning_rate = Config.learning_rate

        if epoch < first_epoch + t_epoch * 0:  # 0-500
            learning_rate = init_learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 500-1000
            learning_rate = init_learning_rate / 10
        elif epoch < first_epoch + t_epoch * 2:  # 1000-1500
            learning_rate = init_learning_rate / 100
        else:  # 1500-2000
            learning_rate = init_learning_rate / 1000
            pass

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass

        return learning_rate

    def load_model(self):
        if os.path.exists(Config.ic_dir):
            self.ic_model.load_state_dict(torch.load(Config.ic_dir))
            Tools.print("load ic model success from {}".format(Config.ic_dir))
        pass

    def train(self):
        Tools.print()
        Tools.print("Training...")

        # Init Update
        try:
            self.ic_model.eval()
            Tools.print("Init label {} .......")
            self.produce_class.reset()
            for image, label, idx in tqdm(self.ic_train_loader):
                image, idx = self.to_cuda(image), self.to_cuda(idx)
                ic_out_logits, ic_out_l2norm = self.ic_model(image)
                self.produce_class.cal_label(ic_out_l2norm, idx)
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2))
        finally:
            pass

        for epoch in range(1, 1 + Config.train_epoch):
            self.ic_model.train()

            Tools.print()
            ic_lr = self.adjust_learning_rate(self.ic_model_optim, epoch)
            Tools.print('Epoch: [{}] ic_lr={}'.format(epoch, ic_lr))

            all_loss = 0.0
            self.produce_class.reset()
            for image, label, idx in tqdm(self.ic_train_loader):
                image, label, idx = self.to_cuda(image), self.to_cuda(label), self.to_cuda(idx)

                ###########################################################################
                # 1 calculate features
                ic_out_logits, ic_out_l2norm = self.ic_model(image)

                # 2 calculate labels
                ic_targets = self.produce_class.get_label(idx)
                self.produce_class.cal_label(ic_out_l2norm, idx)

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
            Tools.print("{:6} loss:{:.3f}".format(epoch, all_loss / len(self.ic_train_loader)))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                self.ic_model.eval()

                val_accuracy = self.test_tool_ic.val(epoch=epoch)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.ic_model.state_dict(), Config.ic_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################


"""
2021-01-19 15:05:33 load ic model success from ../omniglot/ic/2_28_ICConv4_64_2048_1_700_200_100_ic.pkl
2021-01-19 15:05:33 Test 700 .......
2021-01-19 15:05:47 Epoch: 700 Train 0.3937/0.6795 0.0000
2021-01-19 15:05:47 Epoch: 700 Val   0.4567/0.7645 0.0000
2021-01-19 15:05:47 Epoch: 700 Test  0.3743/0.6565 0.0000

2021-01-19 18:06:41 load ic model success from ../omniglot/ic/2_28_ICConv4_64_2048_1_1500_500_200_ic.pkl
2021-01-19 18:06:41 Test 1500 .......
2021-01-19 18:06:48 Epoch: 1500 Train 0.4229/0.7019 0.0000
2021-01-19 18:06:48 Epoch: 1500 Val   0.4881/0.7837 0.0000
2021-01-19 18:06:48 Epoch: 1500 Test  0.3989/0.6843 0.0000

2021-01-19 23:24:03 load ic model success from ../omniglot/ic/3_28_ICConv4_64_2048_1_1500_900_200_ic.pkl
2021-01-19 23:24:03 Test 1500 .......
2021-01-19 23:24:09 Epoch: 1500 Train 0.7669/0.9605 0.0000
2021-01-19 23:24:09 Epoch: 1500 Val   0.8169/0.9616 0.0000
2021-01-19 23:24:09 Epoch: 1500 Test  0.7759/0.9591 0.0000


2021-01-20 03:50:32 load ic model success from ../omniglot/ic/3_28_ICConv4_64_2048_1_1600_1000_300_ic.pkl
2021-01-20 03:50:32 Test 1600 .......
2021-01-20 03:50:41 Epoch: 1600 Train 0.7821/0.9628 0.0000
2021-01-20 03:50:41 Epoch: 1600 Val   0.8235/0.9648 0.0000
2021-01-20 03:50:41 Epoch: 1600 Test  0.8039/0.9667 0.0000


2021-01-20 03:34:55 load ic model success from ../omniglot/ic/2_28_ICConv4_64_1024_1_1600_1000_300_ic.pkl
2021-01-20 03:34:55 Test 1600 .......
2021-01-20 03:35:05 Epoch: 1600 Train 0.8162/0.9727 0.0000
2021-01-20 03:35:05 Epoch: 1600 Val   0.8578/0.9709 0.0000
2021-01-20 03:35:05 Epoch: 1600 Test  0.8486/0.9775 0.0000
"""


##############################################################################################################


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 24
    batch_size = 64
    val_freq = 10
    knn = 20
    ic_ratio = 1

    ##############################################################################################################
    image_size = 28
    # ic_out_dim = 2048
    ic_out_dim = 1024
    ic_net, net_name = EncoderC4(), "ICConv4"

    learning_rate = 0.01
    train_epoch = 1600
    first_epoch, t_epoch = 1000, 300
    adjust_learning_rate = Runner.adjust_learning_rate1
    ##############################################################################################################

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        gpu_id, image_size, net_name, batch_size,
        ic_out_dim, ic_ratio, train_epoch, first_epoch, t_epoch)

    ic_dir = Tools.new_dir("../omniglot/ic/{}_ic.pkl".format(model_name))
    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/omniglot_single'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/omniglot_single'
        if not os.path.isdir(data_root):
            data_root = '/home/ubuntu/Dataset/Partition1/ALISURE/Data/UFSL/omniglot_single'
    else:
        data_root = "F:\\data\\omniglot_single"

    Tools.print(model_name)
    Tools.print(data_root)
    Tools.print(ic_dir)
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    pass
