import os
import math
import torch
import random
import platform
import scipy as sp
import scipy.stats
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
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


class ClassBalancedSamplerTest(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_cl, num_inst, shuffle=True):
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i + j * self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
            pass

        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
                pass
            pass

        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1

    pass


class MiniImageNetTask(object):

    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = dict(zip(class_folders, np.array(range(len(class_folders)))))

        samples = dict()
        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num + test_num]
            pass

        self.train_labels = [labels[os.path.split(x)[0]] for x in self.train_roots]
        self.test_labels = [labels[os.path.split(x)[0]] for x in self.test_roots]
        pass

    pass


class MiniImageNet(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image = Image.open(self.image_roots[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    @staticmethod
    def folders(data_root):
        train_folder = os.path.join(data_root, "train")
        val_folder = os.path.join(data_root, "val")
        test_folder = os.path.join(data_root, "test")

        folders_train = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]
        folders_val = [os.path.join(val_folder, label) for label in os.listdir(val_folder)
                       if os.path.isdir(os.path.join(val_folder, label))]
        folders_test = [os.path.join(test_folder, label) for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))]

        random.seed(1)
        random.shuffle(folders_train)
        random.shuffle(folders_val)
        random.shuffle(folders_test)
        return folders_train, folders_val, folders_test

    @staticmethod
    def get_data_loader(task, num_per_class=1, split='train', sampler_test=False, shuffle=False):
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)), transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #     transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        # transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if split == 'train':
            dataset = MiniImageNet(task, split=split, transform=transform_train)
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            dataset = MiniImageNet(task, split=split, transform=transform_test)
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

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

    pass


##############################################################################################################


class Runner(object):

    def __init__(self, feature_encoder, relation_network):
        self.best_accuracy = 0.0

        self.feature_encoder = feature_encoder
        self.relation_network = relation_network

        # data
        self.folders_train, self.folders_val, self.folders_test = MiniImageNet.folders(Config.data_root)

        # model
        self.feature_encoder.apply(self._weights_init).cuda()
        self.relation_network.apply(self._weights_init).cuda()
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=Config.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, Config.train_episode//3, gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=Config.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, Config.train_episode//3, gamma=0.5)

        self.loss = nn.MSELoss().cuda()
        pass

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
        sample_features = self.feature_encoder(samples.cuda())  # 5x64*19*19
        batch_features = self.feature_encoder(batches.cuda())  # 75x64*19*19

        batch_size, feature_dim, feature_width, feature_height = batch_features.shape

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(Config.num_shot * Config.num_way, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext),
                                   2).view(-1, feature_dim * 2, feature_width, feature_height)

        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, Config.num_way * Config.num_shot)

        return relations

    def train(self):
        Tools.print()
        Tools.print("Training...")

        all_loss = 0.0
        for episode in range(Config.train_episode):
            # init dataset
            task = MiniImageNetTask(self.folders_train, Config.num_way, Config.num_shot, Config.task_batch_size)
            sample_data_loader = MiniImageNet.get_data_loader(task, Config.num_shot, "train", shuffle=False)
            batch_data_loader = MiniImageNet.get_data_loader(task, Config.task_batch_size, split="val", shuffle=True)
            samples, sample_labels = sample_data_loader.__iter__().next()
            batches, batch_labels = batch_data_loader.__iter__().next()

            ###########################################################################
            # calculate features
            relations = self.compare_fsl(samples, batches)
            ###########################################################################

            one_hot_labels = torch.zeros(Config.task_batch_size * Config.num_way,Config.num_way
                                         ).scatter_(1, batch_labels.view(-1, 1), 1).cuda()
            loss = self.loss(relations, one_hot_labels)

            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)

            self.feature_encoder_optim.step()
            self.relation_network_optim.step()
            self.feature_encoder_scheduler.step(episode)
            self.relation_network_scheduler.step(episode)

            all_loss += loss.item()
            if (episode + 1) % Config.print_freq == 0:
                Tools.print("Episode: {} avg loss: {} loss: {} lr: {}".format(
                    episode + 1, all_loss / (episode % Config.val_freq),
                    loss.item(), self.feature_encoder_scheduler.get_lr()))
                pass

            if (episode + 1) % Config.val_freq == 0:
                Tools.print()
                Tools.print("Valing...")
                self.val_train(episode)
                val_accuracy, val_h = self.val_val(episode)
                self.val_test(episode)

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

    def _val(self, folders, sampler_test, all_episode):

        def mean_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
            return m, h

        accuracies = []
        for i in range(all_episode):
            total_rewards = 0
            counter = 0
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本
            task = MiniImageNetTask(folders, Config.num_way, Config.num_shot, Config.task_batch_size)
            sample_data_loader = MiniImageNet.get_data_loader(task, 1, "train", sampler_test=sampler_test, shuffle=False)
            batch_data_loader = MiniImageNet.get_data_loader(task, 3, "val", sampler_test=sampler_test, shuffle=True)
            samples, labels = sample_data_loader.__iter__().next()

            for batches, batch_labels in batch_data_loader:
                ###########################################################################
                # calculate features
                relations = self.compare_fsl(samples, batches)
                ###########################################################################

                _, predict_labels = torch.max(relations.data, 1)
                batch_size = batch_labels.shape[0]
                rewards = [1 if predict_labels[j].cpu() == batch_labels[j] else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)

                counter += batch_size
                pass

            accuracy = total_rewards / 1.0 / counter
            accuracies.append(accuracy)
            pass

        accuracy, h = mean_confidence_interval(accuracies)
        return accuracy, h

    def val_train(self, episode):
        acc, h = self._val(self.folders_train, sampler_test=False, all_episode=Config.test_episode)
        Tools.print("Val {} Train Accuracy: {}".format(episode, acc))
        return acc, h

    def val_val(self, episode):
        acc, h = self._val(self.folders_val, sampler_test=False, all_episode=Config.test_episode)
        Tools.print("Val {} Val Accuracy: {}".format(episode, acc))
        return acc, h

    def val_test(self, episode):
        acc, h = self._val(self.folders_test, sampler_test=False, all_episode=Config.test_episode)
        Tools.print("Val {} Test Accuracy: {}".format(episode, acc))
        return acc, h

    def test(self, test_avg_num=None):
        Tools.print()
        Tools.print("Testing...")
        total_acc = 0.0
        test_avg_num = Config.test_avg_num if test_avg_num is None else test_avg_num
        for episode in range(test_avg_num):
            acc, h = self._val(self.folders_test, sampler_test=True, all_episode=Config.test_episode)
            total_acc += acc
            Tools.print("episode={}, Test accuracy={}, Total accuracy={}".format(episode, acc, total_acc))
            pass

        final_accuracy = total_acc / test_avg_num
        Tools.print("Final accuracy: {}".format(final_accuracy))
        return final_accuracy

    pass


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    train_episode = 300000
    learning_rate = 0.001

    num_way = 5
    num_shot = 1
    task_batch_size = 15

    test_avg_num = 1
    test_episode = 600

    val_freq = 5000  # 5000
    print_freq = 1000

    # model_name = "1"
    model_name = "2"
    _path = "train_one_shot_alisure_new"
    fe_dir = Tools.new_dir("../models/{}/{}_fe_{}way_{}shot.pkl".format(_path, model_name, num_way, num_shot))
    rn_dir = Tools.new_dir("../models/{}/{}_rn_{}way_{}shot.pkl".format(_path, model_name, num_way, num_shot))

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    if model_name == "1": # 0.7547 / 0.4884  - 0.5855 / 0.4600
        feature_encoder, relation_network = CNNEncoder(), RelationNetwork()

    if model_name == "2": # 0.8125 / 0.5215 / 0.5177 - 0.648 / 0.470
        feature_encoder, relation_network = CNNEncoder1(), RelationNetwork1()

    pass


##############################################################################################################


"""
1
2020-10-22 11:14:26 load feature encoder success from ../models/train_one_shot_alisure_new/1_fe_5way_1shot.pkl
2020-10-22 11:14:26 load relation network success from ../models/train_one_shot_alisure_new/1_rn_5way_1shot.pkl
2020-10-22 11:14:48 Val 300000 Train Accuracy: 0.7323333333333333
2020-10-22 11:15:08 Val 300000 Val Accuracy: 0.4902222222222222
2020-10-22 11:16:28 episode=0, Test accuracy=0.504488888888889, Total accuracy=0.504488888888889
2020-10-22 11:17:28 episode=1, Test accuracy=0.49617777777777783, Total accuracy=1.0006666666666668
2020-10-22 11:18:29 episode=2, Test accuracy=0.5102444444444444, Total accuracy=1.510911111111111
2020-10-22 11:19:27 episode=3, Test accuracy=0.49133333333333334, Total accuracy=2.0022444444444445
2020-10-22 11:20:25 episode=4, Test accuracy=0.5007777777777778, Total accuracy=2.5030222222222225
2020-10-22 11:20:25 Final accuracy: 0.5006044444444445

2
2020-10-22 11:23:59 load feature encoder success from ../models/train_one_shot_alisure_new/2_fe_5way_1shot.pkl
2020-10-22 11:23:59 load relation network success from ../models/train_one_shot_alisure_new/2_rn_5way_1shot.pkl
2020-10-22 11:24:18 Val 300000 Train Accuracy: 0.676111111111111
2020-10-22 11:24:38 Val 300000 Val Accuracy: 0.4984444444444444
2020-10-22 11:25:58 episode=0, Test accuracy=0.5174222222222222, Total accuracy=0.5174222222222222
2020-10-22 11:26:59 episode=1, Test accuracy=0.5154888888888888, Total accuracy=1.032911111111111
2020-10-22 11:27:59 episode=2, Test accuracy=0.5238444444444444, Total accuracy=1.5567555555555552
2020-10-22 11:29:00 episode=3, Test accuracy=0.5088222222222223, Total accuracy=2.0655777777777775
2020-10-22 11:30:01 episode=4, Test accuracy=0.5192222222222221, Total accuracy=2.5847999999999995
2020-10-22 11:30:01 Final accuracy: 0.5169599999999999
"""


if __name__ == '__main__':
    runner = Runner(feature_encoder=Config.feature_encoder, relation_network=Config.relation_network)
    # runner.load_model()

    # runner.test()
    # runner.val_train(episode=0)
    # runner.val_val(episode=0)
    # runner.val_test(episode=0)
    # runner.train()

    runner.load_model()
    runner.val_train(episode=Config.train_episode)
    runner.val_val(episode=Config.train_episode)
    runner.val_test(episode=Config.train_episode)
    runner.test(test_avg_num=5)
