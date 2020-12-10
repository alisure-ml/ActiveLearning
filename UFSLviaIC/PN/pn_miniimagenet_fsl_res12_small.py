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
import torch.backends.cudnn as cudnn
from torch.distributions import Bernoulli
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pn_miniimagenet_fsl_test_tool import TestTool
from pn_miniimagenet_tool import ProtoNet, RunnerTool
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


##############################################################################################################


class MiniImageNetDataset(object):

    def __init__(self, data_list, num_way, num_shot):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        self.transform = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), Config.transforms_normalize])
        self.transform_test = transforms.Compose([transforms.ToTensor(), Config.transforms_normalize])
        pass

    def __len__(self):
        return len(self.data_list)

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

    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, now_label, now_image_filename = now_label_image_tuple
        now_label_k_shot_image_tuple = random.sample(self.data_dict[now_label], self.num_shot)

        # 其他样本
        other_label = list(self.data_dict.keys())
        other_label.remove(now_label)
        other_label = random.sample(other_label, self.num_way - 1)
        other_label_k_shot_image_tuple_list = []
        for _label in other_label:
            other_label_k_shot_image_tuple = random.sample(self.data_dict[_label], self.num_shot)
            other_label_k_shot_image_tuple_list.extend(other_label_k_shot_image_tuple)
            pass

        # c_way_k_shot
        c_way_k_shot_tuple_list = now_label_k_shot_image_tuple + other_label_k_shot_image_tuple_list
        random.shuffle(c_way_k_shot_tuple_list)

        task_list = c_way_k_shot_tuple_list + [now_label_image_tuple]
        task_data = torch.cat([torch.unsqueeze(self.read_image(one[2], self.transform), dim=0) for one in task_list])
        task_label = torch.Tensor([int(one_tuple[1] == now_label) for one_tuple in c_way_k_shot_tuple_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    pass


##############################################################################################################


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


class ResNet12(nn.Module):

    def __init__(self, block=BasicBlock, avg_pool=False, drop_rate=0.0):
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


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # all data
        self.data_train = MiniImageNetDataset.get_data_all(Config.data_root)
        self.task_train = MiniImageNetDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, shuffle=True, num_workers=Config.num_workers)

        # model
        self.proto_net = RunnerTool.to_cuda(Config.proto_net)
        RunnerTool.to_cuda(self.proto_net.apply(RunnerTool.weights_init))

        # optim
        self.proto_net_optim = torch.optim.SGD(
            self.proto_net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.test_tool = TestTool(self.proto_test, data_root=Config.data_root,
                                  num_way=Config.num_way_test,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode,
                                  transform=self.task_train.transform_test)
        pass

    def adjust_learning_rate(self, epoch):
        steps = np.sum(epoch > np.asarray(Config.learning_rate_decay_epochs))
        if steps > 0:
            new_lr = Config.learning_rate * (0.1 ** steps)
            for param_group in self.proto_net_optim.param_groups:
                param_group['lr'] = new_lr
            pass
        pass

    def load_model(self):
        if os.path.exists(Config.pn_dir):
            self.proto_net.load_state_dict(torch.load(Config.pn_dir))
            Tools.print("load proto net success from {}".format(Config.pn_dir))
        pass

    def proto(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.proto_net(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        z_support, z_query = z.split(Config.num_shot * Config.num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, Config.num_way, Config.num_shot, z_dim)

        z_support_proto = z_support.mean(2)
        z_query_expand = z_query.expand(z_batch_size, Config.num_way, z_dim)

        dists = torch.pow(z_query_expand - z_support_proto, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def proto_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        sample_z = self.proto_net(samples)  # 5x64*5*5
        batch_z = self.proto_net(batches)  # 75x64*5*5
        sample_z = sample_z.view(num_way, num_shot, -1)
        batch_z = batch_z.view(batch_num, -1)
        _, z_dim = batch_z.shape

        z_proto = sample_z.mean(1)
        z_proto_expand = z_proto.unsqueeze(0).expand(batch_num, num_way, z_dim)
        z_query_expand = batch_z.unsqueeze(1).expand(batch_num, num_way, z_dim)

        dists = torch.pow(z_query_expand - z_proto_expand, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def train(self):
        Tools.print()
        Tools.print("Training...")

        for epoch in range(1, 1 + Config.train_epoch):
            self.proto_net.train()

            Tools.print()
            all_loss = 0.0
            self.adjust_learning_rate(epoch=epoch)
            Tools.print("{:6} lr:{}".format(epoch, self.proto_net_optim.param_groups[0]["lr"]))
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                # 1 calculate features
                log_p_y = self.proto(task_data)

                # 2 loss
                loss = -(log_p_y * task_labels).sum() / task_labels.sum()
                all_loss += loss.item()

                # 3 backward
                self.proto_net.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.proto_net.parameters(), 0.5)
                self.proto_net_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f}".format(epoch, all_loss / len(self.task_train_loader)))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                Tools.print()
                Tools.print("Test {} {} .......".format(epoch, Config.model_name))

                self.proto_net.eval()

                val_accuracy = self.test_tool.val(episode=epoch, is_print=True)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.proto_net.state_dict(), Config.pn_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################


transforms_normalize1 = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))

transforms_normalize2 = transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                             np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    train_epoch = 100
    # learning_rate = 0.05
    learning_rate = 0.01
    # learning_rate = 0.1
    learning_rate_decay_epochs = [50, 80]
    num_workers = 8

    val_freq = 5

    num_way = 10
    # num_way = 5
    # num_way = 3
    # num_way = 2
    num_way_test = 5

    num_shot = 1
    # batch_size = 64
    # batch_size = 32
    batch_size = 16

    episode_size = 15
    test_episode = 600

    is_png = True
    # is_png = False

    block, block_name = BasicBlock, "BasicBlock1"
    proto_net = ResNet12(block=block, avg_pool=True, drop_rate=0.1)

    # transforms_normalize, norm_name = transforms_normalize1, "norm1"
    transforms_normalize, norm_name = transforms_normalize2, "norm2"

    model_name = "{}_{}_{}_{}_{}_{}{}".format(
        gpu_id, train_epoch, batch_size, block_name, learning_rate, norm_name, "_png" if is_png else "")
    Tools.print(model_name)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"
    data_root = os.path.join(data_root, "miniImageNet_png") if is_png else data_root
    Tools.print(data_root)

    pn_dir = Tools.new_dir("../models_pn/fsl_res12_small/{}_pn_{}_{}.pkl".format(model_name, num_way, num_shot))
    pass


##############################################################################################################


"""
0.01
2020-12-03 08:33:17 load proto net success from ../models_pn/fsl_res12/3_150_32_BasicBlock1_0.01_norm2_png_pn_5way_1shot.pkl
2020-12-03 08:36:25 Train 150 Accuracy: 0.9587777777777777
2020-12-03 08:36:25 Val   150 Accuracy: 0.6395555555555555
2020-12-03 08:44:05 episode=150, Mean Test accuracy=0.60076

0.01 2 way 1 shot
2020-12-09 02:42:47 load proto net success from ../models_pn/fsl_res12_small/2_100_64_BasicBlock1_0.01_norm2_png_pn_2_1.pkl
2020-12-09 02:46:15 Train 100 Accuracy: 0.7398888888888889
2020-12-09 02:46:15 Val   100 Accuracy: 0.5752222222222222
2020-12-09 02:55:31 episode=100, Mean Test accuracy=0.5515200000000001

0.01 3 way 1 shot
2020-12-09 04:24:42 load proto net success from ../models_pn/fsl_res12_small/1_100_64_BasicBlock1_0.01_norm2_png_pn_3_1.pkl
2020-12-09 04:28:15 Train 100 Accuracy: 0.7851111111111112
2020-12-09 04:28:15 Val   100 Accuracy: 0.5803333333333334
2020-12-09 04:37:31 episode=100, Mean Test accuracy=0.5670222222222223

0.01 5 way 1 shot
2020-12-09 12:59:56 load proto net success from ../models_pn/fsl_res12_small/2_100_32_BasicBlock1_0.01_norm2_png_pn_5_1.pkl
2020-12-09 13:02:59 Train 100 Accuracy: 0.9440000000000002
2020-12-09 13:02:59 Val   100 Accuracy: 0.6343333333333334
2020-12-09 13:11:04 episode=100, Mean Test accuracy=0.5893244444444445

0.01 10 way 1 shot
2020-12-10 06:06:49 load proto net success from ../models_pn/fsl_res12_small/2_100_16_BasicBlock1_0.01_norm2_png_pn_10_1.pkl
2020-12-10 06:09:16 Train 100 Accuracy: 0.9756666666666668
2020-12-10 06:09:16 Val   100 Accuracy: 0.637888888888889
2020-12-10 06:15:46 episode=100, Mean Test accuracy=0.5988311111111111
"""


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    # runner.proto_net.eval()
    # runner.test_tool.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.proto_net.eval()
    runner.test_tool.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
