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
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from rn_miniimagenet_fsl_test_tool import TestTool
from rn_miniimagenet_ic_test_tool import ICTestTool
from torch.utils.data import DataLoader, Dataset
from rn_miniimagenet_tool import ICModel, ProduceClass, ICModel, CNNEncoder, RelationNetwork, ProduceClass, RunnerTool


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

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                         std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.transform_train_ic = transforms.Compose([
            transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_train_fsl = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

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

        task_data = []
        for one in task_list:
            transform = self.transform_train_ic if one[2] == now_image_filename else self.transform_train_fsl
            task_data.append(torch.unsqueeze(self.read_image(one[2], transform), dim=0))
            pass
        task_data = torch.cat(task_data)

        task_label = torch.Tensor([int(one_tuple[1] == now_label) for one_tuple in c_way_k_shot_tuple_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

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


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # all data
        self.data_train = MiniImageNetDataset.get_data_all(Config.data_root)
        self.task_train = MiniImageNetDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()

        # model
        self.feature_encoder = RunnerTool.to_cuda(Config.feature_encoder)
        self.relation_network = RunnerTool.to_cuda(Config.relation_network)
        self.ic_model = RunnerTool.to_cuda(ICModel(in_dim=Config.ic_in_dim, out_dim=Config.ic_out_dim))

        RunnerTool.to_cuda(self.feature_encoder.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.relation_network.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.ic_model.apply(RunnerTool.weights_init))

        # optim
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=Config.learning_rate)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=Config.learning_rate)
        self.ic_model_optim = torch.optim.Adam(self.ic_model.parameters(), lr=Config.learning_rate)

        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, Config.train_epoch // 3, gamma=0.5)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, Config.train_epoch // 3, gamma=0.5)
        self.ic_model_scheduler = StepLR(self.ic_model_optim, Config.train_epoch // 3, gamma=0.5)

        # loss
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())
        self.fsl_loss = RunnerTool.to_cuda(nn.MSELoss())

        # Eval
        self.test_tool_fsl = TestTool(self.compare_fsl_test, data_root=Config.data_root,
                                      num_way=Config.num_way, num_shot=Config.num_shot,
                                      episode_size=Config.episode_size, test_episode=Config.test_episode,
                                      transform=self.task_train.transform_test)
        self.test_tool_ic = ICTestTool(feature_encoder=self.feature_encoder, ic_model=self.ic_model,
                                       data_root=Config.data_root, batch_size=Config.batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim)
        pass

    def load_model(self):
        if os.path.exists(Config.fe_dir):
            self.feature_encoder.load_state_dict(torch.load(Config.fe_dir))
            Tools.print("load feature encoder success from {}".format(Config.fe_dir))

        if os.path.exists(Config.rn_dir):
            self.relation_network.load_state_dict(torch.load(Config.rn_dir))
            Tools.print("load relation network success from {}".format(Config.rn_dir))

        if os.path.exists(Config.ic_dir):
            self.ic_model.load_state_dict(torch.load(Config.ic_dir))
            Tools.print("load ic model success from {}".format(Config.ic_dir))
        pass

    def compare_fsl(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        fe_inputs = task_data.view([-1, data_num_channel, data_width, data_weight])  # 90, 3, 84, 84

        # feature encoder
        data_features = self.feature_encoder(fe_inputs)  # 90x64*19*19
        _, feature_dim, feature_width, feature_height = data_features.shape

        # calculate
        data_features = data_features.view([-1, data_image_num, feature_dim, feature_width, feature_height])
        data_features_support, data_features_query = data_features.split(Config.num_shot * Config.num_way, dim=1)
        data_features_query_repeat = data_features_query.repeat(1, Config.num_shot * Config.num_way, 1, 1, 1)

        # calculate relations
        relation_pairs = torch.cat((data_features_support, data_features_query_repeat), 2)
        relation_pairs = relation_pairs.view(-1, feature_dim * 2, feature_width, feature_height)

        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, Config.num_way * Config.num_shot)
        return relations, data_features_query.squeeze()

    def compare_fsl_test(self, samples, batches):
        # calculate features
        sample_features = self.feature_encoder(samples)  # 5x64*19*19
        batch_features = self.feature_encoder(batches)  # 75x64*19*19
        batch_size, feature_dim, feature_width, feature_height = batch_features.shape

        # calculate
        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(1).repeat(1, Config.num_shot * Config.num_way, 1, 1, 1)

        # calculate relations
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
        relation_pairs = relation_pairs.view(-1, feature_dim * 2, feature_width, feature_height)

        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, Config.num_way * Config.num_shot)
        return relations

    def train(self):
        Tools.print()
        Tools.print("Training...")

        for epoch in range(Config.train_epoch):
            self.feature_encoder.train()
            self.relation_network.train()
            self.ic_model.train()

            Tools.print()
            self.produce_class.reset()
            all_loss, all_loss_fsl, all_loss_ic = 0.0, 0.0, 0.0
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                relations, query_features = self.compare_fsl(task_data)
                ic_out_logits, ic_out_l2norm = self.ic_model(query_features)

                # 2
                ic_targets = self.produce_class.get_label(ic_labels)
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)

                # 3 loss
                loss_fsl = self.fsl_loss(relations, task_labels) * Config.loss_fsl_ratio
                loss_ic = self.ic_loss(ic_out_logits, ic_targets) * Config.loss_ic_ratio
                loss = loss_fsl + loss_ic
                all_loss += loss.item()
                all_loss_fsl += loss_fsl.item()
                all_loss_ic += loss_ic.item()

                # 4 backward
                self.feature_encoder.zero_grad()
                self.relation_network.zero_grad()
                self.ic_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.ic_model.parameters(), 0.5)
                self.feature_encoder_optim.step()
                self.relation_network_optim.step()
                self.ic_model_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} fsl:{:.3f} ic:{:.3f} lr:{}".format(
                epoch + 1, all_loss / len(self.task_train_loader), all_loss_fsl / len(self.task_train_loader),
                all_loss_ic / len(self.task_train_loader), self.feature_encoder_scheduler.get_last_lr()))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            self.feature_encoder_scheduler.step()
            self.relation_network_scheduler.step()
            self.ic_model_scheduler.step()
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                self.feature_encoder.eval()
                self.relation_network.eval()
                self.ic_model.eval()

                self.test_tool_ic.val(epoch=epoch)
                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True)

                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.feature_encoder.state_dict(), Config.fe_dir)
                    torch.save(self.relation_network.state_dict(), Config.rn_dir)
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
1_900_64_5_1_64_512_1_10.0_0.1_ic
2020-10-25 12:00:32 Test 900 .......
2020-10-25 12:00:55 Epoch: 900 Train 0.4024/0.7240 0.0000
2020-10-25 12:00:55 Epoch: 900 Val   0.5155/0.8940 0.0000
2020-10-25 12:00:55 Epoch: 900 Test  0.4948/0.8785 0.0000
2020-10-25 12:02:21 Train 900 Accuracy: 0.6964444444444444
2020-10-25 12:02:21 Val   900 Accuracy: 0.5076666666666667
2020-10-25 12:05:49 episode=900, Mean Test accuracy=0.49832444444444446

1_600_64_5_1_64_512_1_20.0_0.1_ic_5way_1shot.pkl
2020-10-26 06:32:54 Test 600 .......
2020-10-26 06:33:07 Epoch: 600 Train 0.3914/0.7160 0.0000
2020-10-26 06:33:07 Epoch: 600 Val   0.5058/0.8879 0.0000
2020-10-26 06:33:07 Epoch: 600 Test  0.4782/0.8716 0.0000
2020-10-26 06:35:05 Train 600 Accuracy: 0.6843333333333333
2020-10-26 06:35:05 Val   600 Accuracy: 0.5312222222222223
2020-10-26 06:39:55 episode=600, Mean Test accuracy=0.5076088888888889
"""


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    num_workers = 8
    batch_size = 64
    val_freq = 10

    learning_rate = 0.001

    num_way = 5
    num_shot = 1

    episode_size = 15
    test_episode = 600

    feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
    # feature_encoder, relation_network = CNNEncoder1(), RelationNetwork1()

    # ic
    ic_in_dim = 64
    ic_out_dim = 512
    ic_ratio = 1

    train_epoch = 600

    loss_fsl_ratio = 20.0
    loss_ic_ratio = 0.1

    model_name = "1_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        train_epoch, batch_size, num_way, num_shot, ic_in_dim, ic_out_dim, ic_ratio, loss_fsl_ratio, loss_ic_ratio)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    fe_dir = Tools.new_dir("../models/two_ic_fsl/{}_fe_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    rn_dir = Tools.new_dir("../models/two_ic_fsl/{}_rn_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    ic_dir = Tools.new_dir("../models/two_ic_fsl/{}_ic_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.feature_encoder.eval()
    runner.relation_network.eval()
    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=0, is_print=True)
    runner.test_tool_fsl.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.feature_encoder.eval()
    runner.relation_network.eval()
    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
