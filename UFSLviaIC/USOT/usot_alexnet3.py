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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from usot_fsl_tool import RunnerTool, Normalize
from usot_fsl_test_tool import TestTool


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

        self.transform = transforms.Compose([transforms.Resize(Config.image_size),
                                             transforms.RandomCrop(Config.image_size, padding=8),
                                             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(), Config.transforms_normalize])
        self.transform_test = transforms.Compose([transforms.Resize(Config.image_size),
                                                  transforms.ToTensor(), Config.transforms_normalize])
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


class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    pass


class AlexNetV1(AlexNet):
    output_stride = 8

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 11, 2), nn.BatchNorm2d(96),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 5, 1, groups=2), nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, 3, 1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, 3, 1, groups=2), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, 3, 1, groups=2))
        pass

    pass


class AlexNetV2(nn.Module):
    output_stride = 8

    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.drop_rate = drop_rate
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 3, 1), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 3, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, 3, 1), nn.BatchNorm2d(384), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, 3, 1), nn.BatchNorm2d(384), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, 3, 1, padding=1))
        pass

    def forward(self, x):
        out = self.conv1(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv3(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv4(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv5(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0
        self.adjust_learning_rate = Config.adjust_learning_rate

        # all data
        self.data_train = MiniImageNetDataset.get_data_all(Config.data_root)
        self.task_train = MiniImageNetDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size,
                                            shuffle=True, num_workers=Config.num_workers)

        # model
        self.proto_net = RunnerTool.to_cuda(Config.proto_net)
        RunnerTool.to_cuda(self.proto_net.apply(RunnerTool.weights_init))
        self.norm = Normalize(2)
        self.loss = RunnerTool.to_cuda(nn.MSELoss())

        # optim
        self.proto_net_optim = torch.optim.SGD(
            self.proto_net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.test_tool = TestTool(self.proto_test, data_root=Config.data_root,
                                  num_way=Config.num_way,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode,
                                  transform=self.task_train.transform_test)
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
        out_num, out_channel, out_width, out_weight = net_out.shape
        feature_num = out_channel * out_width * out_weight
        z = net_out.view(data_batch_size, data_image_num, feature_num)

        z_support, z_query = z.split(Config.num_shot * Config.num_way, dim=1)
        z_query = z_query.expand(data_batch_size, Config.num_way, feature_num)

        ######################################################################################################
        z_support = self.norm(z_support)
        similarities = torch.sum(z_support * z_query, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(data_batch_size, Config.num_way, Config.num_shot)
        predicts = torch.sum(similarities, dim=-1)
        ######################################################################################################
        return predicts

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

        z_proto_expand = self.norm(z_proto_expand)
        similarities = torch.sum(z_proto_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, Config.num_way, Config.num_shot)
        predicts = torch.sum(similarities, dim=-1)
        return predicts

    def train(self):
        Tools.print()
        Tools.print("Training...")

        for epoch in range(1, 1 + Config.train_epoch):
            self.proto_net.train()

            Tools.print()
            all_loss = 0.0
            pn_lr = self.adjust_learning_rate(self.proto_net_optim, epoch,
                                              Config.first_epoch, Config.t_epoch, Config.learning_rate)
            Tools.print('Epoch: [{}] pn_lr={}'.format(epoch, pn_lr))

            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                # 1 calculate features
                out = self.proto(task_data)

                # 2 loss
                loss = self.loss(out, task_labels)
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
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    image_size = 127
    num_workers = 8
    val_freq = 5
    episode_size = 15
    test_episode = 600

    num_way = 5
    num_shot = 1
    batch_size = 32

    learning_rate = 0.01
    train_epoch = 160
    first_epoch, t_epoch = 80, 40
    adjust_learning_rate = RunnerTool.adjust_learning_rate2

    ##############################################################################################################
    is_png = True
    # is_png = False

    # proto_net = AlexNetV1()
    proto_net = AlexNetV2()

    # transforms_normalize, norm_name = transforms_normalize1, "norm1"
    transforms_normalize, norm_name = transforms_normalize2, "norm2"
    ##############################################################################################################

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}{}".format(
        gpu_id, train_epoch, batch_size, num_way, num_shot,
        first_epoch, t_epoch, norm_name, "_png" if is_png else "")
    Tools.print(model_name)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"
    data_root = os.path.join(data_root, "miniImageNet_png") if is_png else data_root
    Tools.print(data_root)

    pn_dir = Tools.new_dir("../models_usot/alexnet2/{}.pkl".format(model_name))
    pass


##############################################################################################################


"""
2020-12-09 08:24:21 load proto net success from ../models_usot/alexnet2/3_160_32_5_1_80_40_norm2_png_relu.pkl
2020-12-09 08:25:00 Train 160 Accuracy: 0.7733333333333333
2020-12-09 08:25:58 Val   160 Accuracy: 0.5835555555555555
2020-12-09 08:43:28 episode=160, Test accuracy=0.5645777777777777
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
