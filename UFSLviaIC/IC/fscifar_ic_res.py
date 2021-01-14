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
from torchvision.models import resnet18, resnet34, resnet50, vgg16_bn
sys.path.append("../Common")
from UFSLTool import ResNet12Small, C4Net, RunnerTool


##############################################################################################################


class DatasetIC(Dataset):

    def __init__(self, data_list, image_size=32):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        change = transforms.Resize(image_size) if image_size > 32 else lambda x: x

        self.transform = transforms.Compose([
            change, transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize
        ])
        self.transform_test =  transforms.Compose([change, transforms.ToTensor(), normalize])
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

    def __init__(self, low_dim=512, modify_head=False, resnet=None, vggnet=None, resnet12=None):
        super().__init__()
        self.is_res = True if resnet else False
        self.is_vgg = True if vggnet else False
        self.is_resnet12 = True if resnet12 else False

        if self.is_res:
            self.resnet = resnet(num_classes=low_dim)
            if modify_head:
                self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                pass
        elif self.is_vgg:
            self.vggnet = vggnet()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, low_dim)
            pass
        elif self.is_resnet12:
            self.resnet12 = resnet12()
            self.fc = nn.Linear(512, low_dim)
            pass
        else:
            raise Exception("......")

        self.l2norm = Normalize(2)
        pass

    def forward(self, x, no_last=False):
        if self.is_res:
            out_logits = self.resnet(x)
        elif self.is_vgg:
            features = self.vggnet.features(x)
            features = self.avgpool(features)
            features = torch.flatten(features, 1)
            out_logits = self.fc(features)
            pass
        elif self.is_resnet12:
            features= self.resnet12(x)
            features = torch.flatten(features, 1)

            if no_last:
                return features

            out_logits = self.fc(features)
            pass
        else:
            raise Exception("......")

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
        self.ic_model = self.to_cuda(ICResNet(Config.ic_out_dim, modify_head=Config.modify_head,
                                              resnet=Config.resnet, vggnet=Config.vggnet, resnet12=Config.resnet12))
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
0_32_resnet_34_64_1024_1_1500_300_200_True_ic.pkl
2021-01-10 12:02:50 load ic model success from ../models_CIFARFS/models/ic_res_xx/0_32_resnet_34_64_1024_1_1500_300_200_True_ic.pkl
2021-01-10 12:02:50 Test 1500 .......
2021-01-10 12:03:02 Epoch: 1500 Train 0.6046/0.8583 0.0000
2021-01-10 12:03:02 Epoch: 1500 Val   0.6460/0.9327 0.0000
2021-01-10 12:03:02 Epoch: 1500 Test  0.6775/0.9487 0.0000

1_84_resnet_34_64_1024_1_1500_300_200_True_ic.pkl
2021-01-11 02:14:39 load ic model success from ../models_CIFARFS/models/ic_res_xx/1_84_resnet_34_64_1024_1_1500_300_200_True_ic.pkl
2021-01-11 02:14:39 Test 1500 .......
2021-01-11 02:15:11 Epoch: 1500 Train 0.6009/0.8559 0.0000
2021-01-11 02:15:11 Epoch: 1500 Val   0.6694/0.9426 0.0000
2021-01-11 02:15:11 Epoch: 1500 Test  0.6913/0.9553 0.0000


2_CIFARFS_32_resnet_50_64_256_1_1500_300_200_True_ic.pkl
2021-01-13 10:10:05 load ic model success from ../models_CIFARFS/models/ic_res_xx/2_CIFARFS_32_resnet_50_64_256_1_1500_300_200_True_ic.pkl
2021-01-13 10:10:05 Test 1500 .......
2021-01-13 10:10:22 Epoch: 1500 Train 0.5903/0.8597 0.0000
2021-01-13 10:10:22 Epoch: 1500 Val   0.6291/0.9320 0.0000
2021-01-13 10:10:22 Epoch: 1500 Test  0.6629/0.9491 0.0000

1_CIFARFS_32_resnet_34_64_256_1_1500_300_200_True_ic.pkl
2021-01-13 05:39:58 load ic model success from ../models_CIFARFS/models/ic_res_xx/1_CIFARFS_32_resnet_34_64_256_1_1500_300_200_True_ic.pkl
2021-01-13 05:39:58 Test 1500 .......
2021-01-13 05:40:14 Epoch: 1500 Train 0.5920/0.8573 0.0000
2021-01-13 05:40:14 Epoch: 1500 Val   0.6267/0.9305 0.0000
2021-01-13 05:40:14 Epoch: 1500 Test  0.6554/0.9477 0.0000

"""


"""
1_FC100_32_resnet_34_64_512_1_1500_300_200_True_ic.pkl
2021-01-11 23:46:47 load ic model success from ../models_CIFARFS/models/ic_res_xx/1_FC100_32_resnet_34_64_512_1_1500_300_200_True_ic.pkl
2021-01-11 23:46:47 Test 1500 .......
2021-01-11 23:47:02 Epoch: 1500 Train 0.6312/0.9010 0.0000
2021-01-11 23:47:02 Epoch: 1500 Val   0.4491/0.8022 0.0000
2021-01-11 23:47:02 Epoch: 1500 Test  0.4330/0.8199 0.0000

2021-01-11 23:27:50 load proto net success from ../models_CIFARFS/models/ic_res_xx/1_FC100_32_resnet_34_64_512_1_1500_300_200_True_ic.pkl
2021-01-11 23:28:32 Train 400 Accuracy: 0.6137777777777778
2021-01-11 23:29:17 Val   400 Accuracy: 0.328
2021-01-11 23:30:00 Test1 400 Accuracy: 0.3562222222222223
2021-01-11 23:33:04 Test2 400 Accuracy: 0.3567111111111111
2021-01-11 23:48:20 episode=400, Test accuracy=0.35066666666666674
2021-01-11 23:48:20 episode=400, Test accuracy=0.3503555555555556
2021-01-11 23:48:20 episode=400, Test accuracy=0.3482
2021-01-11 23:48:20 episode=400, Test accuracy=0.3518444444444444
2021-01-11 23:48:20 episode=400, Test accuracy=0.3502222222222222
2021-01-11 23:48:20 episode=400, Mean Test accuracy=0.3502577777777778


2_FC100_32_resnet12small_64_512_1_1500_300_200_True_ic.pkl
2021-01-13 00:01:58 load ic model success from ../models_CIFARFS/models/ic_res_xx/2_FC100_32_resnet12small_64_512_1_1500_300_200_True_ic.pkl
2021-01-13 00:01:58 Test 1500 .......
2021-01-13 00:02:09 Epoch: 1500 Train 0.6300/0.9076 0.0000
2021-01-13 00:02:09 Epoch: 1500 Val   0.4515/0.8102 0.0000
2021-01-13 00:02:09 Epoch: 1500 Test  0.4428/0.8276 0.0000

2021-01-13 00:18:16 load proto net success from ../models_CIFARFS/models/ic_res_xx/2_FC100_32_resnet12small_64_512_1_1500_300_200_True_ic.pkl
2021-01-13 00:18:42 Train 400 Accuracy: 0.614
2021-01-13 00:19:09 Val   400 Accuracy: 0.347
2021-01-13 00:19:35 Test1 400 Accuracy: 0.34933333333333333
2021-01-13 00:21:13 Test2 400 Accuracy: 0.3578222222222222
2021-01-13 00:29:17 episode=400, Test accuracy=0.3554222222222222
2021-01-13 00:29:17 episode=400, Test accuracy=0.3564888888888889
2021-01-13 00:29:17 episode=400, Test accuracy=0.3528
2021-01-13 00:29:17 episode=400, Test accuracy=0.3572666666666667
2021-01-13 00:29:17 episode=400, Test accuracy=0.35715555555555556
2021-01-13 00:29:17 episode=400, Mean Test accuracy=0.3558266666666666



0_FC100_32_resnet_50_64_256_1_1500_300_200_True_ic.pkl
2021-01-13 07:21:37 load ic model success from ../models_CIFARFS/models/ic_res_xx/0_FC100_32_resnet_50_64_256_1_1500_300_200_True_ic.pkl
2021-01-13 07:21:37 Test 1500 .......
2021-01-13 07:21:54 Epoch: 1500 Train 0.6161/0.9056 0.0000
2021-01-13 07:21:54 Epoch: 1500 Val   0.4420/0.7968 0.0000
2021-01-13 07:21:54 Epoch: 1500 Test  0.4233/0.8142 0.0000

"""


##############################################################################################################


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8
    batch_size = 64
    val_freq = 10
    knn = 100
    ic_ratio = 1

    ##############################################################################################################
    # dataset_name = "CIFARFS"
    dataset_name = "FC100"

    # image_size = 84
    image_size = 32

    # ic_out_dim = 1024
    ic_out_dim = 512
    # ic_out_dim = 256

    resnet, vggnet, resnet12 = None, None, None
    # resnet, net_name = resnet18, "resnet_18"
    # resnet, net_name = resnet34, "resnet_34"
    # resnet, net_name = resnet50, "resnet_50"
    resnet12, net_name = ResNet12Small, "resnet12small"

    # modify_head = False
    modify_head = True
    ##############################################################################################################

    ##############################################################################################################
    learning_rate = 0.01
    train_epoch = 1500
    first_epoch, t_epoch = 300, 200
    adjust_learning_rate = Runner.adjust_learning_rate1
    ##############################################################################################################

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        gpu_id, dataset_name, image_size, net_name, batch_size,
        ic_out_dim, ic_ratio, train_epoch, first_epoch, t_epoch, modify_head)

    ic_dir = Tools.new_dir("../models_CIFARFS/models/ic_res_xx/{}_ic.pkl".format(model_name))
    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/{}'.format(dataset_name)
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/{}'.format(dataset_name)
    else:
        data_root = "F:\\data\\{}".format(dataset_name)

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
