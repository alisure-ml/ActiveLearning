import os
import time
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
import torchvision.transforms as transforms
from miniimagenet_ic_test_tool import ICTestTool
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50, vgg16_bn


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


class ICNet(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out_logits = self.net(x)
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


class MyConv4Net(nn.Module):

    def __init__(self, hid_dim=64, z_dim=64):
        super().__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 41
        self.conv_block_2 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 21
        self.conv_block_3 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 10
        self.conv_block_4 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 5
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hid_dim, z_dim)
        pass

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out_logits = self.fc(out)
        return out_logits

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class MyVGGNet(nn.Module):

    def __init__(self, low_dim=512, vggnet=None):
        super().__init__()
        self.vggnet = vggnet()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, low_dim)
        pass

    def forward(self, x):
        features = self.vggnet.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        out_logits = self.fc(features)
        return out_logits

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class MyResNet(nn.Module):

    def __init__(self, low_dim=512, modify_head=False, resnet=None):
        super().__init__()
        self.resnet = resnet(num_classes=low_dim)
        if modify_head:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        pass

    def forward(self, x):
        out_logits = self.resnet(x)
        return out_logits

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


##############################################################################################################


class Runner(object):

    def __init__(self, config):
        self.config = config
        self.best_accuracy = 0.0
        self.adjust_learning_rate = self.config.adjust_learning_rate

        # data
        self.data_train = MiniImageNetIC.get_data_all(self.config.data_root)
        self.ic_train_loader = DataLoader(MiniImageNetIC(self.data_train),
                                          self.config.batch_size, True, num_workers=self.config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), self.config.ic_out_dim, self.config.ic_ratio)
        self.produce_class.init()

        # model
        self.ic_model = self.to_cuda(self.config.ic_net)
        self.ic_loss = self.to_cuda(nn.CrossEntropyLoss())

        self.ic_model_optim = torch.optim.SGD(self.ic_model.parameters(),
                                              lr=self.config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # Eval
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
                                       data_root=self.config.data_root, batch_size=self.config.batch_size,
                                       num_workers=self.config.num_workers, ic_out_dim=self.config.ic_out_dim)
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

    def adjust_learning_rate1(self, optimizer, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=self.config.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = self.config.t_epoch
        first_epoch = self.config.first_epoch
        init_learning_rate = self.config.learning_rate
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

    def adjust_learning_rate2(self, optimizer, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=self.config.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = self.config.t_epoch
        first_epoch = self.config.first_epoch
        init_learning_rate = self.config.learning_rate

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
        if os.path.exists(self.config.ic_dir):
            self.ic_model.load_state_dict(torch.load(self.config.ic_dir))
            Tools.print("load ic model success from {}".format(self.config.ic_dir), txt_path=self.config.log_file)
        pass

    def train(self):
        Tools.print()
        Tools.print("Training...", txt_path=self.config.log_file)

        # Init Update
        try:
            self.ic_model.eval()
            Tools.print("Init label {} .......", txt_path=self.config.log_file)
            self.produce_class.reset()
            for image, label, idx in tqdm(self.ic_train_loader):
                image, idx = self.to_cuda(image), self.to_cuda(idx)
                ic_out_logits, ic_out_l2norm = self.ic_model(image)
                self.produce_class.cal_label(ic_out_l2norm, idx)
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count,
                                              self.produce_class.count_2), txt_path=self.config.log_file)
        finally:
            pass

        for epoch in range(1, 1 + self.config.train_epoch):
            self.ic_model.train()

            Tools.print()
            ic_lr = self.adjust_learning_rate(self, self.ic_model_optim, epoch)
            Tools.print('Epoch: [{}] ic_lr={}'.format(epoch, ic_lr), txt_path=self.config.log_file)

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
            Tools.print("{:6} loss:{:.3f}".format(
                epoch, all_loss / len(self.ic_train_loader)), txt_path=self.config.log_file)
            Tools.print("Train: [{}] {}/{}".format(
                epoch, self.produce_class.count, self.produce_class.count_2), txt_path=self.config.log_file)
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % self.config.val_freq == 0:
                self.ic_model.eval()

                val_accuracy = self.test_tool_ic.val(epoch=epoch, txt_path=self.config.log_file)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.ic_model.state_dict(), self.config.ic_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch), txt_path=self.config.log_file)
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################

"""

"""

##############################################################################################################


class Config(object):

    net_name_conv4 = "conv4"
    net_name_vgg16_bn = "vgg16_bn"
    net_name_res18 = "res18"
    net_name_res34 = "res34"
    net_name_res50 = "res50"

    def __init__(self, gpu_id, ic_out_dim=512, modify_head=False, net_name=net_name_conv4):
        self.gpu_id = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.num_workers = 8
        self.batch_size = 64
        self.val_freq = 10

        # ic
        self.ic_out_dim = ic_out_dim
        self.ic_ratio = 1

        self.modify_head = modify_head

        self.net_name = net_name
        self.net = self.get_net(self.net_name)
        self.ic_net = ICNet(net=self.net)

        self.is_png = True
        # self.is_png = False

        self.learning_rate = 0.01
        self.train_epoch = 1700
        self.first_epoch, self.t_epoch = 500, 200
        self.adjust_learning_rate = Runner.adjust_learning_rate1

        self.model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}{}{}".format(
            self.gpu_id, self.net_name, self.batch_size, self.ic_out_dim, self.ic_ratio,
            self.train_epoch, self.first_epoch, self.t_epoch, self.learning_rate,
            "_head" if self.modify_head else "", "_png" if self.is_png else "")

        self.time = Tools.get_format_time()
        self.ic_dir = Tools.new_dir("../models_abl/ic_res_xx/{}_{}_ic.pkl".format(self.time, self.model_name))
        self.log_file = self.ic_dir.replace(".pkl", ".txt")
        self.data_root = self.get_data_root()
        Tools.print(self.model_name, txt_path=self.log_file)
        Tools.print(self.data_root, txt_path=self.log_file)
        pass

    def get_data_root(self):
        if "Linux" in platform.platform():
            data_root = '/mnt/4T/Data/data/miniImagenet'
            if not os.path.isdir(data_root):
                data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
        else:
            data_root = "F:\\data\\miniImagenet"
        data_root = os.path.join(data_root, "miniImageNet_png") if self.is_png else data_root
        return data_root

    def get_net(self, net_name):
        if net_name == "conv4":
            net = MyConv4Net(hid_dim=64, z_dim=self.ic_out_dim)
        elif net_name == "vgg16_bn":
            net = MyVGGNet(low_dim=self.ic_out_dim, vggnet=vgg16_bn)
        elif net_name == "res18":
            net = MyResNet(low_dim=self.ic_out_dim, modify_head=self.modify_head, resnet=resnet18)
        elif net_name == "res34":
            net = MyResNet(low_dim=self.ic_out_dim, modify_head=self.modify_head, resnet=resnet34)
        elif net_name == "res50":
            net = MyResNet(low_dim=self.ic_out_dim, modify_head=self.modify_head, resnet=resnet50)
        else:
            raise Exception("....................")
        return net

    pass


if __name__ == '__main__':
    ic_out_dim_list = [64, 128, 256, 512, 1024, 2048]
    net_name_list = [Config.net_name_conv4, Config.net_name_vgg16_bn,
                     Config.net_name_res18, Config.net_name_res34, Config.net_name_res50]

    _gpu_id = 0
    now_net_name = net_name_list[2]
    modify_head = False
    for _ic_out_dim in ic_out_dim_list:
        _config = Config(gpu_id=_gpu_id, ic_out_dim=_ic_out_dim, modify_head=modify_head, net_name=now_net_name)

        runner = Runner(config=_config)
        runner.ic_model.eval()
        runner.test_tool_ic.val(epoch=0, is_print=True, txt_path=_config.log_file)

        runner.train()

        runner.load_model()
        runner.ic_model.eval()
        runner.test_tool_ic.val(epoch=_config.train_epoch, is_print=True, txt_path=_config.log_file)
        pass
    pass
