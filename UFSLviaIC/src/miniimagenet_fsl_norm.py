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
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


##############################################################################################################


class MiniImageNetDataset(object):

    def __init__(self, data_list, is_train, num_way, num_shot):
        self.data_list, self.is_train = data_list, is_train
        self.num_way, self.num_shot = num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass
        normalize = transforms.Normalize(mean=Config.MEAN_PIXEL, std=Config.STD_PIXEL)
        self.transform_train = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform = self.transform_train if self.is_train else self.transform_test
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
        self.data_train, self.data_val, self.data_test = MiniImageNetDataset.get_data_all(Config.data_root)

        self.task_train = MiniImageNetDataset(self.data_train, True, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, shuffle=True, num_workers=Config.num_workers)

        self.task_test_train = MiniImageNetDataset(self.data_train, False, Config.num_way, Config.num_shot)
        self.task_test_val = MiniImageNetDataset(self.data_val, False, Config.num_way, Config.num_shot)
        self.task_test_test = MiniImageNetDataset(self.data_test, False, Config.num_way, Config.num_shot)
        self.task_test_train_loader = DataLoader(self.task_test_train, Config.batch_size, shuffle=False, num_workers=Config.num_workers)
        self.task_test_val_loader = DataLoader(self.task_test_val, Config.batch_size, shuffle=False, num_workers=Config.num_workers)
        self.task_test_test_loader = DataLoader(self.task_test_test, Config.batch_size, shuffle=False, num_workers=Config.num_workers)

        # model
        self.feature_encoder = cuda(CNNEncoder())
        self.relation_network = cuda(RelationNetwork())
        self.feature_encoder.apply(self._weights_init).cuda()
        self.relation_network.apply(self._weights_init).cuda()

        # loss
        self.fsl_loss = cuda(nn.MSELoss())

        # optim
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=Config.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, Config.train_epoch//3, gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=Config.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, Config.train_epoch//3, gamma=0.5)
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

    def compare_fsl(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        fe_inputs = task_data.view([-1, data_num_channel, data_width, data_weight])  # 90, 3, 84, 84

        # feature encoder
        data_features = self.feature_encoder(fe_inputs)  # 90x64*19*19
        _, feature_dim, feature_width, feature_height = data_features.shape
        data_features = data_features.view([-1, data_image_num, feature_dim, feature_width, feature_height])
        data_features_support, data_features_query = data_features.split(Config.num_shot * Config.num_way, dim=1)
        data_features_query_repeat = data_features_query.repeat(1, Config.num_shot * Config.num_way, 1, 1, 1)

        # calculate relations
        relation_pairs = torch.cat((data_features_support, data_features_query_repeat), 2)
        relation_pairs = relation_pairs.view(-1, feature_dim * 2, feature_width, feature_height)
        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, Config.num_way * Config.num_shot)

        return relations, data_features_query.squeeze()

    def train(self):
        Tools.print()
        Tools.print("Training...")

        for epoch in range(Config.train_epoch):
            self.feature_encoder.train()
            self.relation_network.train()

            Tools.print()
            all_loss = 0.0
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = cuda(task_data), cuda(task_labels)

                # 1 calculate features
                relations, query_features = self.compare_fsl(task_data)

                # 2 loss
                loss = self.fsl_loss(relations, task_labels)
                all_loss += loss.item()

                # 3 backward
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
                self.val_fsl(epoch, self.task_test_train_loader, name="Train")
                val_accuracy = self.val_fsl(epoch, self.task_test_val_loader, name="Val")
                self.val_fsl(epoch, loader=self.task_test_test_loader, name="Test")
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

    def val_fsl(self, epoch, loader, name=None):
        accuracies = []
        total_rewards, counter = 0, 0
        for task_data, task_labels, _ in loader:
            task_data, task_labels = cuda(task_data), cuda(task_labels)
            batch_size = task_labels.shape[0]

            relations, _ = self.compare_fsl(task_data)
            _, predict_labels = torch.max(relations.data, 1)
            rewards = [int(task_labels[i][predict_labels[i]]) for i in range(batch_size)]

            total_rewards += np.sum(rewards)
            counter += batch_size
            accuracies.append(total_rewards / 1.0 / counter)
            pass

        final_accuracy = np.mean(np.array(accuracies, dtype=np.float))
        Tools.print("Val {} {} Accuracy: {}".format(epoch, name, final_accuracy))
        return final_accuracy

    def val_fsl_test(self, epoch, test_avg_num=2):
        Tools.print()
        Tools.print("Testing...")
        total_accuracy = 0.0
        for _which in range(test_avg_num):
            test_accuracy = self.val_fsl(epoch, loader=self.task_test_test_loader, name="Test {}".format(_which))
            total_accuracy += test_accuracy
            pass
        final_accuracy = total_accuracy / Config.test_avg_num
        Tools.print("Final Test accuracy: {}".format(final_accuracy))
        return final_accuracy

    pass


##############################################################################################################


"""
2020-10-22 16:58:21 load feature encoder success from ../models/fsl/1_64_5_1_fe_5way_1shot.pkl
2020-10-22 16:58:21 load relation network success from ../models/fsl/1_64_5_1_rn_5way_1shot.pkl
2020-10-22 16:59:16 Val 1000 Final Train Accuracy: 0.7856667234316946
2020-10-22 16:59:29 Val 1000 Final Val Accuracy: 0.5062298349603024
2020-10-22 16:59:46 Val 1000 Test 0 Accuracy: 0.49566093146157125
"""


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_epoch = 600
    learning_rate = 0.001
    num_workers = 8

    val_freq = 1

    num_way = 5
    num_shot = 1
    batch_size = 64
    test_avg_num = 2

    # norm = "1"
    norm = "2"
    if norm == "1":
        MEAN_PIXEL = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        STD_PIXEL = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
    else:
        MEAN_PIXEL = [0.92206, 0.92206, 0.92206]
        STD_PIXEL = [0.08426, 0.08426, 0.08426]
        pass

    model_name = "2_{}_{}_{}_{}".format(batch_size, num_way, num_shot, norm)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    fe_dir = Tools.new_dir("../models/fsl2/{}_fe_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    rn_dir = Tools.new_dir("../models/fsl2/{}_rn_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.feature_encoder.eval()
    runner.relation_network.eval()
    runner.val_fsl(epoch=0, loader=runner.task_test_train_loader, name="First Train")
    runner.val_fsl(epoch=0, loader=runner.task_test_val_loader, name="First Val")
    runner.val_fsl(epoch=0, loader=runner.task_test_test_loader, name="First Test")

    runner.train()

    runner.load_model()
    runner.feature_encoder.eval()
    runner.relation_network.eval()
    runner.val_fsl(epoch=Config.train_epoch, loader=runner.task_test_train_loader, name="Final Train")
    runner.val_fsl(epoch=Config.train_epoch, loader=runner.task_test_val_loader, name="Final Val")
    runner.val_fsl_test(epoch=Config.train_epoch, test_avg_num=Config.test_avg_num)
    pass
