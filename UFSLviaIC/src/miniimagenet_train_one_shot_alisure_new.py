import os
import math
import torch
import random
import platform
import numpy as np
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


class ClassBalancedSampler(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
            pass

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
            pass

        return iter(batch)

    def __len__(self):
        return 1

    pass


class MiniImageNet(Dataset):

    def __init__(self, labels, image_roots):
        self.labels = labels
        self.image_roots = image_roots

        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image = Image.open(self.image_roots[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    @staticmethod
    def get_task(character_folders, num_classes, train_num, test_num):
        class_folders = random.sample(character_folders, num_classes)
        labels = dict(zip(class_folders, np.array(range(len(class_folders)))))

        samples = dict()
        train_roots, test_roots = [], []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            train_roots += samples[c][:train_num]
            test_roots += samples[c][train_num:train_num + test_num]
            pass

        train_labels = [labels[os.path.split(x)[0]] for x in train_roots]
        test_labels = [labels[os.path.split(x)[0]] for x in test_roots]
        return train_roots, test_roots, train_labels, test_labels

    @staticmethod
    def folders(data_root):
        train_folder = os.path.join(data_root, "train")
        folders_train = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]
        random.shuffle(folders_train)
        return folders_train

    pass


##############################################################################################################


# Original
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


# Original --> MaxPool, FC input
class CNNEncoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
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


class RelationNetwork1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 5 * 5, 64)  # 64
        self.fc2 = nn.Linear(64, 1)  # 64
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

    def __init__(self, feature_encoder, relation_network):
        self.best_accuracy = 0.0

        self.feature_encoder = self.to_cuda(feature_encoder)
        self.relation_network = self.to_cuda(relation_network)
        self.to_cuda(self.feature_encoder.apply(self._weights_init))
        self.to_cuda(self.relation_network.apply(self._weights_init))

        # model
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=Config.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, Config.train_episode // 3, gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=Config.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, Config.train_episode // 3, gamma=0.5)

        self.loss = self.to_cuda(nn.MSELoss())

        # data
        self.folders_train = MiniImageNet.folders(Config.data_root)

        # Test
        self.test_tool = TestTool(self.compare_fsl, data_root=Config.data_root,
                                  num_way=Config.num_way,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode)
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

    def train(self):
        Tools.print()
        Tools.print("Training...")

        all_loss = 0.0
        for episode in range(Config.train_episode):
            # init dataset
            train_roots, test_roots, train_labels, test_labels = MiniImageNet.get_task(
                self.folders_train, Config.num_way, Config.num_shot, Config.episode_size)

            sample_dataset = MiniImageNet(train_labels, train_roots)
            sample_sampler = ClassBalancedSampler(Config.num_shot, Config.num_way, Config.num_shot, shuffle=False)
            sample_data_loader = DataLoader(sample_dataset, Config.num_shot * Config.num_way, sampler=sample_sampler)

            batch_dataset = MiniImageNet(test_labels, test_roots)
            batch_sampler = ClassBalancedSampler(Config.episode_size, Config.num_way, Config.episode_size, shuffle=True)
            batch_data_loader = DataLoader(batch_dataset, Config.episode_size * Config.num_way, sampler=batch_sampler)

            samples, sample_labels = sample_data_loader.__iter__().next()
            batches, batch_labels = batch_data_loader.__iter__().next()

            ###########################################################################
            # calculate features
            samples, batches = self.to_cuda(samples), self.to_cuda(batches)
            relations = self.compare_fsl(samples, batches)
            ###########################################################################

            one_hot_labels = self.to_cuda(torch.zeros(Config.episode_size * Config.num_way,
                                                      Config.num_way).scatter_(1, batch_labels.long().view(-1, 1), 1))
            loss = self.loss(relations, one_hot_labels)

            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)

            self.feature_encoder_optim.step()
            self.relation_network_optim.step()
            self.feature_encoder_scheduler.step()
            self.relation_network_scheduler.step()

            all_loss += loss.item()
            if (episode + 1) % Config.print_freq == 0:
                Tools.print("Episode: {} avg loss: {} loss: {} lr: {}".format(
                    episode + 1, all_loss / (episode % Config.val_freq),
                    loss.item(), self.feature_encoder_scheduler.get_last_lr()))
                pass

            if (episode + 1) % Config.val_freq == 0:
                Tools.print()
                Tools.print("Test ...")
                val_accuracy = self.test_tool.val(episode=episode, is_print=True)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.feature_encoder.state_dict(), Config.fe_dir)
                    torch.save(self.relation_network.state_dict(), Config.rn_dir)
                    Tools.print("Save networks for episode: {}".format(episode))
                    pass

                all_loss = 0.0
                Tools.print()
                pass

            pass

        pass

    pass


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    train_episode = 300000
    learning_rate = 0.001

    num_way = 5
    num_shot = 1
    episode_size = 15

    test_avg_num = 1
    test_episode = 600

    val_freq = 5000  # 5000
    print_freq = 1000

    model_name = "1"
    # model_name = "2"
    _path = "train_one_shot_alisure_new"
    fe_dir = Tools.new_dir("../models/{}/{}_fe_{}way_{}shot.pkl".format(_path, model_name, num_way, num_shot))
    rn_dir = Tools.new_dir("../models/{}/{}_rn_{}way_{}shot.pkl".format(_path, model_name, num_way, num_shot))

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    if model_name == "1":  # 0.7547 / 0.4884  - 0.5855 / 0.4600
        feature_encoder, relation_network = CNNEncoder(), RelationNetwork()

    if model_name == "2":  # 0.8125 / 0.5215 / 0.5177 - 0.648 / 0.470
        feature_encoder, relation_network = CNNEncoder1(), RelationNetwork1()

    pass


##############################################################################################################


"""
"""


if __name__ == '__main__':
    runner = Runner(feature_encoder=Config.feature_encoder, relation_network=Config.relation_network)
    # runner.load_model()

    runner.test_tool.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.test_tool.val(episode=Config.train_episode, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_episode, is_print=True)
    pass
