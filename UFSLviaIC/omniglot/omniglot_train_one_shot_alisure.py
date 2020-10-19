import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from alisuretool.Tools import Tools
import task_generator_alisure as tg
from torch.optim.lr_scheduler import StepLR


class CNNEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=0),
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

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        pass

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

    pass


class Runner(object):

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.feature_dim = 64
        self.relation_dim = 8
        self.class_num = 20
        self.sample_num_per_class = 1
        self.batch_num_per_class = 10
        self.episode = 1000000
        self.test_episode = 1000
        self.learning_rate = 0.001
        self.hidden_unit = 10

        self.print_freq = 100
        self.test_freq = 5000

        self.feature_encoder_dir = "./models/omniglot_feature_encoder_{}way_{}shot.pkl".format(
            self.class_num, self.sample_num_per_class)
        self.relation_network_dir = "./models/omniglot_relation_network_{}way_{}shot.pkl".format(
            self.class_num, self.sample_num_per_class)

        # data
        self.metatrain_character_folders, self.metatest_character_folders = tg.omniglot_character_folders()

        # model
        Tools.print("init neural networks")
        self.feature_encoder = CNNEncoder().to(self.device)
        self.relation_network = RelationNetwork(self.feature_dim, self.relation_dim).to(self.device)

        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=self.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, step_size=100000, gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=self.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, step_size=100000, gamma=0.5)

        self.loss = nn.MSELoss().to(self.device)
        pass

    def load_model(self):
        if os.path.exists(self.feature_encoder_dir):
            self.feature_encoder.load_state_dict(torch.load(self.feature_encoder_dir))
            Tools.print("load feature encoder success")
            pass

        if os.path.exists(self.relation_network_dir):
            self.relation_network.load_state_dict(torch.load(self.relation_network_dir))
            Tools.print("load relation network success")
        pass

    def train(self):
        Tools.print("Training...")

        last_accuracy = 0.0
        for step in range(self.episode):
            degrees = random.choice([0, 90, 180, 270])

            task = tg.OmniglotTask(self.metatrain_character_folders, self.class_num,
                                   self.sample_num_per_class, self.batch_num_per_class)
            sample_data_loader = tg.get_data_loader(task, num_per_class=self.sample_num_per_class,
                                                   split="train", shuffle=False, rotation=degrees)
            batch_data_loader = tg.get_data_loader(task, num_per_class=self.batch_num_per_class,
                                                  split="test", shuffle=True, rotation=degrees)
            # sample datas
            samples, sample_labels = sample_data_loader.__iter__().next()
            batches, batch_labels = batch_data_loader.__iter__().next()

            # calculate features
            sample_features = self.feature_encoder(samples.to(self.device))  # 5x64*5*5
            batch_features = self.feature_encoder(batches.to(self.device))  # 20x64*5*5

            # calculate relations
            sample_features_ext = sample_features.unsqueeze(0).repeat(
                self.batch_num_per_class * self.class_num, 1, 1, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(
                self.sample_num_per_class * self.class_num, 1, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

            relation_pairs = torch.cat((sample_features_ext,
                                        batch_features_ext), 2).view(-1, self.feature_dim * 2, 5, 5)
            relations = self.relation_network(relation_pairs).view(-1, self.class_num)

            one_hot_labels = torch.zeros(self.batch_num_per_class * self.class_num, self.class_num)
            one_hot_labels = one_hot_labels.scatter(1, batch_labels.long().view(-1, 1), 1).to(self.device)
            loss = self.loss(relations, one_hot_labels)

            # training
            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)

            self.feature_encoder_optim.step()
            self.relation_network_optim.step()
            self.feature_encoder_scheduler.step(step)
            self.relation_network_scheduler.step(step)

            if (step + 1) % self.print_freq == 0:
                Tools.print("step: {}, loss {}".format(step + 1, loss.item()))
                pass

            if (step + 1) % self.test_freq == 0:
                # test
                Tools.print("Testing...")
                test_accuracy = self.test()
                Tools.print("test accuracy: {}".format(test_accuracy))

                if test_accuracy > last_accuracy:
                    # save networks
                    torch.save(self.feature_encoder.state_dict(), self.feature_encoder_dir)
                    torch.save(self.relation_network.state_dict(), self.relation_network_dir)
                    Tools.print("save networks for step: {}".format(step))
                    last_accuracy = test_accuracy
                    pass
                pass
            pass
        pass

    def test(self):
        total_rewards = 0
        for i in range(self.test_episode):
            degrees = random.choice([0, 90, 180, 270])
            task = tg.OmniglotTask(self.metatest_character_folders, self.class_num,
                                   self.sample_num_per_class, self.batch_num_per_class)
            sample_dataloader = tg.get_data_loader(task, num_per_class=self.sample_num_per_class,
                                                   split="train", shuffle=False, rotation=degrees)
            test_dataloader = tg.get_data_loader(task, num_per_class=self.sample_num_per_class,
                                                 split="test", shuffle=True, rotation=degrees)

            sample_images, sample_labels = sample_dataloader.__iter__().next()
            test_images, test_labels = test_dataloader.__iter__().next()

            # calculate features
            sample_features = self.feature_encoder(sample_images.to(self.device))  # 5x64
            test_features = self.feature_encoder(test_images.to(self.device))  # 20x64

            # calculate relations
            sample_features_ext = sample_features.unsqueeze(0).repeat(
                self.sample_num_per_class * self.class_num, 1, 1, 1, 1)
            test_features_ext = test_features.unsqueeze(0).repeat(
                self.sample_num_per_class * self.class_num, 1, 1, 1, 1)
            test_features_ext = torch.transpose(test_features_ext, 0, 1)

            relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1,
                                                                                         self.feature_dim * 2, 5, 5)
            relations = self.relation_network(relation_pairs).view(-1, self.class_num)

            _, predict_labels = torch.max(relations.data, 1)

            rewards = [1 if predict_labels[j].cpu() == test_labels[j] else 0 for j in range(self.class_num)]
            total_rewards += np.sum(rewards)
            pass

        test_accuracy = total_rewards / 1.0 / self.class_num / self.test_episode
        return test_accuracy

    pass


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    runner = Runner()
    runner.load_model()
    # runner.test()
    runner.train()
    pass
