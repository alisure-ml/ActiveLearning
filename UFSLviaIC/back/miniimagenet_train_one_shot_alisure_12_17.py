import os
import math
import torch
import random
import scipy as sp
import scipy.stats
import numpy as np
import torch.nn as nn
from PIL import Image
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
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
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
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
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
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    @staticmethod
    def mini_imagenet_folders(train_folder, val_folder, test_folder):
        metatrain_folders = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
                             if os.path.isdir(os.path.join(train_folder, label))]
        metaval_folders = [os.path.join(val_folder, label) for label in os.listdir(val_folder)
                            if os.path.isdir(os.path.join(val_folder, label))]
        metatest_folders = [os.path.join(test_folder, label) for label in os.listdir(test_folder)
                            if os.path.isdir(os.path.join(test_folder, label))]
        random.seed(1)
        random.shuffle(metatrain_folders)
        random.shuffle(metaval_folders)
        random.shuffle(metatest_folders)
        return metatrain_folders, metaval_folders, metatest_folders

    @staticmethod
    def get_mini_imagenet_data_loader(task, num_per_class=1, split='train', shuffle=False):
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        dataset = MiniImageNet(task, split=split, transform=transforms.Compose([transforms.ToTensor(), normalize]))

        if split == 'train':
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        elif split == "val":
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
        else:  # test
            sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
            pass

        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

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
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = out.view(out.size(0),-1)
        return out  # 64

    pass


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size * 3 * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        pass

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.feature_dim = 64
        self.relation_dim = 8
        self.hidden_unit = 10

        self.class_num = 5
        self.sample_num_per_class = 1
        self.batch_num_per_class = 15

        self.train_episode = 1000  # 500000
        self.val_episode = 600
        self.test_avg_num = 10
        self.test_episode = 600

        self.learning_rate = 0.001

        self.print_freq = 100
        self.val_freq = 500  # 5000

        self.feature_encoder_dir = Tools.new_dir("../models/miniimagenet_feature_encoder_{}way_{}shot.pkl".format(
            self.class_num, self.sample_num_per_class))
        self.relation_network_dir = Tools.new_dir("../models/miniimagenet_relation_network_{}way_{}shot.pkl".format(
            self.class_num, self.sample_num_per_class))

        # data
        self.metatrain_folders, self.metaval_folders, self.metatest_folders = MiniImageNet.mini_imagenet_folders(
            train_folder='/mnt/4T/Data/miniImagenet/train',
            val_folder='/mnt/4T/Data/miniImagenet/val', test_folder='/mnt/4T/Data/miniImagenet/test')

        # model
        (self.feature_encoder, self.relation_network, self.feature_encoder_scheduler,
         self.relation_network_scheduler, self.feature_encoder_optim, self.relation_network_optim) = self._model()

        self.loss = self._loss()
        pass

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())
            pass
        pass

    def _model(self):
        Tools.print("Init neural networks")

        feature_encoder = CNNEncoder()
        relation_network = RelationNetwork(self.feature_dim, self.relation_dim)

        feature_encoder.apply(self._weights_init)
        relation_network.apply(self._weights_init)

        feature_encoder.cuda()
        relation_network.cuda()

        feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=self.learning_rate)
        feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
        relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=self.learning_rate)
        relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

        return (feature_encoder, relation_network, feature_encoder_scheduler,
                relation_network_scheduler, feature_encoder_optim, relation_network_optim)

    @staticmethod
    def _loss():
        mse = nn.MSELoss().cuda()
        return mse

    def _val(self, meta_folders, split, episode):

        def mean_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
            return m, h

        accuracies = []
        for i in range(episode):
            total_rewards = 0
            counter = 0
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本
            task = MiniImageNetTask(meta_folders, self.class_num, self.sample_num_per_class, self.batch_num_per_class)
            sample_dataloader = MiniImageNet.get_mini_imagenet_data_loader(task, 1, split="train", shuffle=False)
            batch_dataloader = MiniImageNet.get_mini_imagenet_data_loader(task, 3, split=split, shuffle=True)
            sample_images, sample_labels = sample_dataloader.__iter__().next()

            for batch_images, batch_labels in batch_dataloader:
                ###########################################################################
                # calculate features
                relations = self._compare_fsl(sample_images, batch_images)
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

    def load_model(self):
        if os.path.exists(self.feature_encoder_dir):
            self.feature_encoder.load_state_dict(torch.load(self.feature_encoder_dir))
            # self.feature_encoder.load_state_dict(torch.load(self.feature_encoder_dir, map_location='cuda:0'))
            Tools.print("load feature encoder success from {}".format(self.feature_encoder_dir))

        if os.path.exists(self.relation_network_dir):
            self.relation_network.load_state_dict(torch.load(self.relation_network_dir))
            # self.relation_network.load_state_dict(torch.load(self.relation_network_dir, map_location='cuda:0'))
            Tools.print("load relation network success from {}".format(self.relation_network_dir))
        pass

    def _compare_fsl(self, samples, batches):
        # calculate features
        sample_features = self.feature_encoder(samples.cuda())  # 5x64*19*19
        batch_features = self.feature_encoder(batches.cuda())  # 75x64*19*19
        batch_size, _, feature_width, feature_height = batch_features.shape

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(
            self.sample_num_per_class * self.class_num, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext),
                                   2).view(-1, self.feature_dim * 2, feature_width, feature_height)
        relations = self.relation_network(relation_pairs).view(-1, self.class_num * self.sample_num_per_class)
        return relations

    def train(self):
        Tools.print()
        Tools.print("Training...")

        last_accuracy = 0.0
        for episode in range(self.train_episode):
            # init dataset
            task = MiniImageNetTask(self.metatrain_folders, self.class_num,
                                    self.sample_num_per_class, self.batch_num_per_class)
            sample_dataloader = MiniImageNet.get_mini_imagenet_data_loader(
                task, num_per_class=self.sample_num_per_class, split="train", shuffle=False)
            batch_dataloader = MiniImageNet.get_mini_imagenet_data_loader(
                task, num_per_class=self.batch_num_per_class, split="val", shuffle=True)
            samples, sample_labels = sample_dataloader.__iter__().next()
            batches, batch_labels = batch_dataloader.__iter__().next()

            ###########################################################################
            # calculate features
            relations = self._compare_fsl(samples, batches)
            ###########################################################################

            one_hot_labels = torch.zeros(self.batch_num_per_class * self.class_num,
                                         self.class_num).scatter_(1, batch_labels.view(-1, 1), 1).cuda()
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

            if (episode + 1) % self.print_freq == 0:
                Tools.print("Episode: {} loss: {}".format(episode + 1, loss.item()))
                pass

            if episode % self.val_freq == 0:
                Tools.print()
                Tools.print("Valing...")
                val_accuracy, h = self.val(is_print=True)
                if val_accuracy > last_accuracy:
                    # save networks
                    torch.save(self.feature_encoder.state_dict(), self.feature_encoder_dir)
                    torch.save(self.relation_network.state_dict(), self.relation_network_dir)
                    Tools.print("Save networks for episode: {}".format(episode))
                    last_accuracy = val_accuracy
                    pass

                pass

            pass

        pass

    def val(self, is_print=False):
        val_accuracy, h = self._val(self.metaval_folders, split="val", episode=self.val_episode)
        if is_print:
            Tools.print()
            Tools.print("Val Accuracy: {} h: {}".format(val_accuracy, h))
            pass
        return val_accuracy, h

    def test(self, is_print=True):
        Tools.print()
        Tools.print("Testing...")
        total_accuracy = 0.0
        for episode in range(self.test_avg_num):
            test_accuracy, h = self._val(self.metatest_folders, split="test", episode=self.test_episode)
            total_accuracy += test_accuracy
            Tools.print("episode={}, Test accuracy={}, H={}, Total accuracy={}".format(episode, test_accuracy,
                                                                                       h, total_accuracy))
            pass

        final_accuracy = total_accuracy / self.test_avg_num
        if is_print:
            Tools.print("Final accuracy: {}".format(final_accuracy))
            pass
        return final_accuracy

    pass


##############################################################################################################


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    runner = Runner()
    runner.load_model()
    runner.test(is_print=True)
    runner.val(is_print=True)
    # runner.train()
    runner.val(is_print=True)
    runner.test(is_print=True)
