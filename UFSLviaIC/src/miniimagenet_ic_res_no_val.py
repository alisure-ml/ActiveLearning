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
from miniimagenet_ic_test_tool import ICTestTool
from torch.utils.data import DataLoader, Dataset


##############################################################################################################


class MiniImageNetIC(Dataset):

    def __init__(self, data_list, image_size=84):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        norm = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                    std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = image if self.transform is None else self.transform(image)
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


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0
        self.adjust_learning_rate = Config.adjust_learning_rate

        # data
        self.data_train = MiniImageNetIC.get_data_all(Config.data_root)
        self.ic_train_loader = DataLoader(MiniImageNetIC(self.data_train),
                                          Config.batch_size, True, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()

        # model
        self.ic_model = self.to_cuda(ICResNet(Config.ic_out_dim))
        self.ic_loss = self.to_cuda(nn.CrossEntropyLoss())

        # self.ic_model_optim = torch.optim.Adam(self.ic_model.parameters(), lr=Config.learning_rate)
        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # Eval
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
                                       data_root=Config.data_root, batch_size=Config.batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim)
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
        else:  # 1000-1500
            learning_rate = init_learning_rate / 100
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

        for epoch in range(Config.train_epoch):
            if Config.is_set_eval_train:
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
            Tools.print("{:6} loss:{:.3f}".format(epoch + 1, all_loss / len(self.ic_train_loader)))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                if Config.is_set_eval_train:
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
2_64_128_2_0.001 no stand resnet18
2020-10-21 08:12:53 Test 830 .......
2020-10-21 10:19:59 load ic model success from ../models/ic/2_64_128_2_0.001_ic.pkl
2020-10-21 10:20:15 Epoch: [1000] Final Train 0.3792/0.6978
2020-10-21 10:20:19 Epoch: [1000] Final Val 0.5069/0.8831
2020-10-21 10:20:24 Epoch: [1000] Final Test 0.4816/0.8652

2_64_128_2_0.001 stand resnet18
2020-10-22 05:41:30   1000 loss:1.109 lr:0.001
2020-10-22 05:41:30 Train: [999] 12085/2720
2020-10-22 05:41:30 load ic model success from ../models/ic/2_64_128_2_0.001_ic.pkl
2020-10-22 05:41:41 Epoch: [1000] Final Train 0.3528/0.6679
2020-10-22 05:41:44 Epoch: [1000] Final Val 0.4795/0.8682
2020-10-22 05:41:48 Epoch: [1000] Final Test 0.4552/0.8438

2_64_512_2_0.001 stand resnet18
2020-10-22 06:17:58   1000 loss:1.244 lr:0.001
2020-10-22 06:17:58 Train: [999] 10926/1860
2020-10-22 06:17:58 load ic model success from ../models/ic/2_64_512_2_0.001_ic.pkl
2020-10-22 06:18:05 Epoch: [1000] Final Train 0.3662/0.6720
2020-10-22 06:18:08 Epoch: [1000] Final Val 0.4821/0.8674
2020-10-22 06:18:10 Epoch: [1000] Final Test 0.4570/0.8373

1_64_512_2_200_100_0.01 stand resnet18
2020-10-22 20:33:53 Train: [999] 13294/2094
2020-10-22 20:33:53 load ic model success from ../models/ic/1_64_512_2_200_100_0.01_ic.pkl
2020-10-22 20:34:05 Epoch: [1000] Final Train 0.4563/0.7607
2020-10-22 20:34:09 Epoch: [1000] Final Val 0.5566/0.9118
2020-10-22 20:34:13 Epoch: [1000] Final Test 0.5407/0.8958

1_64_512_2_500_500_0.01 stand resnet18
2020-10-23 01:40:50 Train: [1499] 11522/1793
2020-10-23 01:40:50 load ic model success from ../models/ic/1_64_512_2_500_500_0.01_ic.pkl
2020-10-23 01:40:59 Epoch: [1500] Final Train 0.4722/0.7703
2020-10-23 01:41:01 Epoch: [1500] Final Val 0.5675/0.9091
2020-10-23 01:41:04 Epoch: [1500] Final Test 0.5453/0.9009

1_64_512_1_500_200_0.01 stand resnet18
2020-10-23 12:43:06 Train: [2099] 8892/1647
2020-10-23 12:43:06 load ic model success from ../models/ic_res/1_64_512_1_500_200_0.01_ic.pkl
2020-10-23 12:43:14 Epoch: [2100] Final Train 0.4959/0.7840
2020-10-23 12:43:17 Epoch: [2100] Final Val 0.5796/0.9132
2020-10-23 12:43:19 Epoch: [2100] Final Test 0.5544/0.9008

1_64_512_2_500_200_0.01 stand resnet18
2020-10-23 09:10:26 Train: [2099] 8755/1740
2020-10-23 09:10:26 load ic model success from ../models/ic_res/1_64_512_2_500_200_0.01_ic.pkl
2020-10-23 09:10:35 Epoch: [2100] Final Train 0.4848/0.7825
2020-10-23 09:10:37 Epoch: [2100] Final Val 0.5739/0.9129
2020-10-23 09:10:40 Epoch: [2100] Final Test 0.5520/0.9032

1_64_512_3_500_200_0.01 stand resnet18
2020-10-23 09:10:49 Train: [2099] 8481/1567
2020-10-23 09:10:49 load ic model success from ../models/ic_res/1_64_512_3_500_200_0.01_ic.pkl
2020-10-23 09:10:56 Epoch: [2100] Final Train 0.4790/0.7811
2020-10-23 09:10:58 Epoch: [2100] Final Val 0.5698/0.9137
2020-10-23 09:11:00 Epoch: [2100] Final Test 0.5463/0.9018

1_256_512_1_500_200_0.01 stand resnet18
2020-10-23 20:00:52 Train: [2099] 7335/1629
2020-10-23 20:00:52 load ic model success from ../models/ic_res/1_256_512_1_500_200_0.01_ic.pkl
2020-10-23 20:00:59 Epoch: [2100] Final Train 0.4366/0.7411
2020-10-23 20:01:01 Epoch: [2100] Final Val 0.5459/0.8941
2020-10-23 20:01:03 Epoch: [2100] Final Test 0.5059/0.8737

1_64_512_1_500_200_0.01_2 stand resnet18 norm2
2020-10-23 23:15:57 Train: [2099] 9191/1837
2020-10-23 23:15:58 load ic model success from ../models/ic_res/1_64_512_1_500_200_0.01_2_ic.pkl
2020-10-23 23:16:05 Epoch: [2100] Final Train 0.5005/0.7878
2020-10-23 23:16:07 Epoch: [2100] Final Val 0.5781/0.9140
2020-10-23 23:16:09 Epoch: [2100] Final Test 0.5600/0.9048
"""


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_workers = 8
    # batch_size = 32
    batch_size = 64
    # batch_size = 256
    val_freq = 10

    learning_rate = 0.01

    # train_epoch = 1000
    # first_epoch, t_epoch = 200, 100
    # adjust_learning_rate = Runner.adjust_learning_rate1

    # train_epoch = 1500
    # first_epoch, t_epoch = 500, 500
    # adjust_learning_rate = Runner.adjust_learning_rate2

    train_epoch = 2100
    first_epoch, t_epoch = 500, 200
    adjust_learning_rate = Runner.adjust_learning_rate1

    # ic
    ic_out_dim = 512
    ic_ratio = 1
    # ic_ratio = 2
    # ic_ratio = 3

    # is_set_eval_train = True
    is_set_eval_train = False

    model_name = "1_{}_{}_{}_{}_{}_{}_{}".format(
        batch_size, ic_out_dim, ic_ratio, first_epoch, t_epoch, learning_rate, is_set_eval_train)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    ic_dir = Tools.new_dir("../models/ic_res_no_val/{}_ic.pkl".format(model_name))
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    if Config.is_set_eval_train:
        runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=0, is_print=True)

    runner.train()

    runner.load_model()
    if Config.is_set_eval_train:
        runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    pass
