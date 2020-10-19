import os
import math
import torch
import scipy as sp
import scipy.stats
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from alisuretool.Tools import Tools
import task_generator_alisure as tg
from torch.optim.lr_scheduler import StepLR


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
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
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

    pass


class Runner(object):

    def __init__(self):
        self.feature_dim = 64
        self.relation_dim = 8
        self.hidden_unit = 10

        self.class_num = 5
        self.sample_num_per_class = 1
        self.batch_num_per_class = 15

        self.episode = 500000
        self.test_episode = 600

        self.learning_rate = 0.001

        self.print_freq = 100
        self.test_freq = 5000

        self.feature_encoder_dir = "./models/miniimagenet_feature_encoder_{}way_{}shot.pkl".format(
            self.class_num, self.sample_num_per_class)
        self.relation_network_dir = "./models/miniimagenet_relation_network_{}way_{}shot.pkl".format(
            self.class_num, self.sample_num_per_class)

        # data
        self.metatrain_folders, self.metatest_folders = tg.mini_imagenet_folders()

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
        # Step 2: init neural networks
        Tools.print("init neural networks")

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

    def load_model(self):
        if os.path.exists(self.feature_encoder_dir):
            self.feature_encoder.load_state_dict(torch.load(self.feature_encoder_dir, map_location='cuda:0'))
            Tools.print("load feature encoder success")

        if os.path.exists(self.relation_network_dir):
            self.relation_network.load_state_dict(torch.load(self.relation_network_dir, map_location='cuda:0'))
            Tools.print("load relation network success")
        pass

    def test(self, is_print=False):

        def mean_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
            return m, h

        accuracies = []
        for i in range(self.test_episode):
            total_rewards = 0
            counter = 0
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本
            task = tg.MiniImagenetTask(self.metatest_folders, self.class_num,
                                       self.sample_num_per_class, self.batch_num_per_class)
            sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=1, split="train", shuffle=False)
            test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=3, split="test", shuffle=True)
            sample_images, sample_labels = sample_dataloader.__iter__().next()

            for test_images, test_labels in test_dataloader:
                # calculate features
                sample_features = self.feature_encoder(sample_images.cuda())  # 5x64x19x19
                test_features = self.feature_encoder(test_images.cuda())  # 15x64x19x19

                batch_size, _, feature_width, feature_height = test_features.shape
                counter += batch_size

                # calculate relations
                sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(self.sample_num_per_class * self.class_num,
                                                                      1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)
                relation_pairs = torch.cat((sample_features_ext, test_features_ext),
                                           2).view(-1,  self.feature_dim * 2, feature_width, feature_height)
                relations = self.relation_network(relation_pairs).view(-1, self.class_num)
                _, predict_labels = torch.max(relations.data, 1)

                rewards = [1 if predict_labels[j].cpu() == test_labels[j] else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)
                pass

            accuracy = total_rewards / 1.0 / counter
            accuracies.append(accuracy)
            pass

        test_accuracy, h = mean_confidence_interval(accuracies)

        if is_print:
            Tools.print("test accuracy: {} h: {}".format(test_accuracy, h))
            pass

        return test_accuracy, h

    def train(self):
        Tools.print("Training...")

        last_accuracy = 0.0
        for episode in range(self.episode):
            self.feature_encoder_scheduler.step(episode)
            self.relation_network_scheduler.step(episode)

            # init dataset
            task = tg.MiniImagenetTask(self.metatrain_folders, self.class_num,
                                       self.sample_num_per_class, self.batch_num_per_class)
            sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=self.sample_num_per_class,
                                                                 split="train", shuffle=False)
            batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=self.batch_num_per_class,
                                                                split="test", shuffle=True)

            # sample datas
            samples, sample_labels = sample_dataloader.__iter__().next()
            batches, batch_labels = batch_dataloader.__iter__().next()

            # calculate features
            sample_features = self.feature_encoder(samples.cuda())  # 5x64*19*19
            batch_features = self.feature_encoder(batches.cuda())  # 75x64*19*19

            batch_size, _, feature_width, feature_height = batch_features.shape

            # calculate relations
            sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(self.sample_num_per_class * self.class_num,
                                                                    1, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
            relation_pairs = torch.cat((sample_features_ext, batch_features_ext),
                                       2).view(-1, self.feature_dim * 2, feature_width, feature_height)
            relations = self.relation_network(relation_pairs).view(-1, self.class_num * self.sample_num_per_class)

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

            if (episode + 1) % self.print_freq == 0:
                Tools.print("episode: {} loss: {}".format(episode + 1, loss.item()))
                pass

            if episode % self.test_freq == 0:
                # test
                Tools.print("Testing...")
                test_accuracy, h = self.test(is_print=True)

                if test_accuracy > last_accuracy:
                    # save networks
                    torch.save(self.feature_encoder.state_dict(), self.feature_encoder_dir)
                    torch.save(self.relation_network.state_dict(), self.relation_network_dir)
                    Tools.print("save networks for episode:", episode)
                    last_accuracy = test_accuracy
                    pass

                pass

            pass

        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    runner = Runner()
    runner.load_model()
    runner.test(is_print=True)
    runner.train()
    runner.test(is_print=True)
