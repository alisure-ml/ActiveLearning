import os
import math
import torch
import random
import platform
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from miniimagenet_fsl_test_tool import TestTool
from torch.utils.data import DataLoader, Dataset


##############################################################################################################


class MiniImageNetDataset(object):

    def __init__(self, data_list, num_way, num_shot, episode_size=15):
        self.data_list, self.num_way, self.num_shot, self.episode_size = data_list, num_way, num_shot, episode_size

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

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

    def __getitem__(self, item):
        # 每类数量
        each_class_num = self.num_shot + self.episode_size

        # Train image
        train_tuple_list, test_tuple_list = [], []

        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, now_label, now_image_filename = now_label_image_tuple
        now_label_all_image_tuple = random.sample(self.data_dict[now_label], each_class_num)

        train_tuple_list.append(now_label_all_image_tuple[0])
        test_tuple_list.extend(now_label_all_image_tuple[1:])

        # 其他样本
        other_label = list(self.data_dict.keys())
        other_label.remove(now_label)
        other_label = random.sample(other_label, self.num_way - 1)
        for _label in other_label:
            other_label_image_tuple = random.sample(self.data_dict[_label], each_class_num)
            train_tuple_list.append(other_label_image_tuple[0])
            test_tuple_list.extend(other_label_image_tuple[1:])
            pass

        # shuffle
        random.shuffle(train_tuple_list)
        random.shuffle(test_tuple_list)

        train_data_list = torch.cat([torch.unsqueeze(self.read_image(one[2], self.transform), dim=0) for one in train_tuple_list])
        test_data_list = torch.cat([torch.unsqueeze(self.read_image(one[2], self.transform), dim=0) for one in test_tuple_list])
        test_label_list = torch.Tensor([[int(one_test[1] == one_train[1]) for one_train in train_tuple_list] for one_test in test_tuple_list])
        train_index_list = torch.Tensor([one[0] for one in train_tuple_list]).long()
        test_index_list = torch.Tensor([one[0] for one in test_tuple_list]).long()
        return train_data_list, test_data_list, test_label_list

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

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


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # all data
        self.data_train = MiniImageNetDataset.get_data_all(Config.data_root)
        self.task_train = MiniImageNetDataset(self.data_train, Config.num_way,
                                              Config.num_shot, episode_size=Config.episode_size)
        self.task_train_loader = DataLoader(self.task_train, batch_size=Config.batch_size,
                                            shuffle=True, num_workers=Config.num_workers)

        # model
        self.feature_encoder = self.to_cuda(CNNEncoder())
        self.relation_network = self.to_cuda(RelationNetwork())
        self.to_cuda(self.feature_encoder.apply(self._weights_init))
        self.to_cuda(self.relation_network.apply(self._weights_init))

        # optim
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=Config.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, Config.train_epoch // 3, gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=Config.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, Config.train_epoch // 3, gamma=0.5)

        # loss
        self.loss = self.to_cuda(nn.MSELoss())

        self.test_tool = TestTool(self.compare_fsl, data_root=Config.data_root,
                                  num_way=Config.num_way,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode,
                                  transform=self.task_train.transform_test)
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
            m.bias.data = torch.ones(m.bias.data.size())
            pass
        pass

    def load_model(self):
        if os.path.exists(Config.fe_dir):
            self.feature_encoder.load_state_dict(torch.load(Config.fe_dir))
            Tools.print("load feature encoder success from {}".format(Config.fe_dir))

        if os.path.exists(Config.rn_dir):
            self.relation_network.load_state_dict(torch.load(Config.rn_dir))
            Tools.print("load relation network success from {}".format(Config.rn_dir))
        pass

    def compare_fsl(self, samples, batches):
        # calculate features
        sample_features = self.feature_encoder(samples)  # 5x64*19*19
        batch_features = self.feature_encoder(batches)  # 75x64*19*19
        batch_size, feature_dim, feature_width, feature_height = batch_features.shape

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(
            Config.num_shot * Config.num_way, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext,
                                    batch_features_ext), 2).view(-1, feature_dim * 2, feature_width, feature_height)

        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, Config.num_way * Config.num_shot)
        return relations

    def compare_fsl_batch(self, samples, batches):
        batch_size, num_samples, channel, height, width = samples.shape
        num_batches = batches.shape[1]

        samples = samples.view(-1, channel, height, width)
        batches = batches.view(-1, channel, height, width)

        # calculate features
        sample_features = self.feature_encoder(samples)  # 5x64*19*19
        batch_features = self.feature_encoder(batches)  # 75x64*19*19
        _, feature_dim, feature_width, feature_height = batch_features.shape
        sample_features = sample_features.view(batch_size, -1, feature_dim, feature_width, feature_height)
        batch_features = batch_features.view(batch_size, -1, feature_dim, feature_width, feature_height)


        # calculate relations
        sample_features_ext = sample_features.unsqueeze(1).repeat(1, num_batches, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(2).repeat(1, 1, num_samples, 1, 1, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 3).view(-1, feature_dim * 2, feature_width, feature_height)

        relations = self.relation_network(relation_pairs)
        relations = relations.view(batch_size, -1, num_samples)
        return relations

    def train(self):
        Tools.print()
        Tools.print("Training...")

        for epoch in range(Config.train_epoch):
            self.feature_encoder.train()
            self.relation_network.train()

            Tools.print()
            all_loss = 0.0
            for train_data_list, test_data_list, test_label_list in tqdm(self.task_train_loader):
                batch_labels = self.to_cuda(test_label_list)
                samples, batches = self.to_cuda(train_data_list), self.to_cuda(test_data_list)
                ###########################################################################
                samples = torch.squeeze(samples, dim=0)
                batches = torch.squeeze(batches, dim=0)
                batch_labels = torch.squeeze(batch_labels, dim=0)

                compare_fsl = self.compare_fsl if Config.batch_size == 1 else self.compare_fsl_batch
                relations = compare_fsl(samples, batches)

                loss = self.loss(relations, batch_labels)
                all_loss += loss.item()

                self.feature_encoder.zero_grad()
                self.relation_network.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)
                self.feature_encoder_optim.step()
                self.relation_network_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} lr:{}".format(
                epoch + 1, all_loss / len(self.task_train_loader), self.feature_encoder_scheduler.get_last_lr()))

            self.feature_encoder_scheduler.step()
            self.relation_network_scheduler.step()
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                self.feature_encoder.eval()
                self.relation_network.eval()

                Tools.print()
                Tools.print("Test {} .......".format(epoch))
                val_accuracy = self.test_tool.val(episode=epoch, is_print=True)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.feature_encoder.state_dict(), Config.fe_dir)
                    torch.save(self.relation_network.state_dict(), Config.rn_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train_epoch = 15
    learning_rate = 0.001
    num_workers = 8

    num_way = 5
    num_shot = 1
    episode_size = 15
    batch_size = 1

    test_avg_num = 1
    test_episode = 600

    val_freq = 1

    model_name = "1"
    _path = "train_alisure_simple_new"
    fe_dir = Tools.new_dir("../models/{}/{}_fe_{}way_{}shot.pkl".format(_path, model_name, num_way, num_shot))
    rn_dir = Tools.new_dir("../models/{}/{}_rn_{}way_{}shot.pkl".format(_path, model_name, num_way, num_shot))

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pass


##############################################################################################################


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.test_tool.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.test_tool.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
