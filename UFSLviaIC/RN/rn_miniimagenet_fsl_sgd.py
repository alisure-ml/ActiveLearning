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
from rn_miniimagenet_fsl_test_tool import TestTool
from torch.utils.data import DataLoader, Dataset
from rn_miniimagenet_tool import CNNEncoder, RelationNetwork, CNNEncoder1, RelationNetwork1, RunnerTool


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
        self.transform = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
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


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0
        self.adjust_learning_rate = Config.adjust_learning_rate

        # all data
        self.data_train = MiniImageNetDataset.get_data_all(Config.data_root)
        self.task_train = MiniImageNetDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, shuffle=True, num_workers=Config.num_workers)

        # model
        self.feature_encoder = RunnerTool.to_cuda(Config.feature_encoder)
        self.relation_network = RunnerTool.to_cuda(Config.relation_network)
        RunnerTool.to_cuda(self.feature_encoder.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.relation_network.apply(RunnerTool.weights_init))

        # optim
        self.feature_encoder_optim = torch.optim.SGD(
            self.feature_encoder.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.relation_network_optim = torch.optim.SGD(
            self.relation_network.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # loss
        self.loss = RunnerTool.to_cuda(nn.MSELoss())

        self.test_tool = TestTool(self.compare_fsl_test, data_root=Config.data_root,
                                  num_way=Config.num_way,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode,
                                  transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.fe_dir):
            self.feature_encoder.load_state_dict(torch.load(Config.fe_dir))
            Tools.print("load feature encoder success from {}".format(Config.fe_dir))

        if os.path.exists(Config.rn_dir):
            self.relation_network.load_state_dict(torch.load(Config.rn_dir))
            Tools.print("load relation network success from {}".format(Config.rn_dir))

        Tools.print("load model over")
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
        return relations

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

            Tools.print()
            fe_lr= self.adjust_learning_rate(self.feature_encoder_optim, epoch,
                                             Config.first_epoch, Config.t_epoch, Config.learning_rate)
            rn_lr = self.adjust_learning_rate(self.relation_network_optim, epoch,
                                              Config.first_epoch, Config.t_epoch, Config.learning_rate)
            Tools.print('Epoch: [{}] fe_lr={} rn_lr={}'.format(epoch, fe_lr, rn_lr))

            all_loss = 0.0
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                # 1 calculate features
                relations = self.compare_fsl(task_data)

                # 2 loss
                loss = self.loss(relations, task_labels)
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
            Tools.print("{:6} loss:{:.3f}".format(epoch + 1, all_loss / len(self.task_train_loader)))
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    learning_rate = 0.01
    num_workers = 8

    num_way = 5
    num_shot = 1
    batch_size = 64

    val_freq = 10
    episode_size = 15
    test_episode = 600

    train_epoch = 400
    first_epoch, t_epoch = 200, 100
    adjust_learning_rate = RunnerTool.adjust_learning_rate2

    feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
    # feature_encoder, relation_network = CNNEncoder1(), RelationNetwork1()

    model_name = "2_{}_{}_{}_{}_{}_{}".format(train_epoch, batch_size, num_way, num_shot, first_epoch, t_epoch)
    # model_name = "2_{}_{}_{}_{}".format(train_epoch, batch_size, num_way, num_shot)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    fe_dir = Tools.new_dir("../models/fsl_sgd/{}_fe_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    rn_dir = Tools.new_dir("../models/fsl_sgd/{}_rn_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    pass


##############################################################################################################


"""
1 feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
2020-10-24 18:26:20 load feature encoder success from ../models/fsl_sgd/1_600_64_5_1_fe_5way_1shot.pkl
2020-10-24 18:26:20 load relation network success from ../models/fsl_sgd/1_600_64_5_1_rn_5way_1shot.pkl
2020-10-24 18:28:17 Train 600 Accuracy: 0.7347777777777778
2020-10-24 18:28:17 Val   600 Accuracy: 0.5202222222222223
2020-10-24 18:32:35 episode=600, Mean Test accuracy=0.49725333333333327

2 feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
2020-10-25 10:09:40 load feature encoder success from ../models/fsl_sgd/2_400_64_5_1_200_100_fe_5way_1shot.pkl
2020-10-25 10:09:41 load relation network success from ../models/fsl_sgd/2_400_64_5_1_200_100_rn_5way_1shot.pkl
2020-10-25 10:11:13 Train 400 Accuracy: 0.7486666666666667
2020-10-25 10:11:13 Val   400 Accuracy: 0.5175555555555555
2020-10-25 10:15:04 episode=400, Mean Test accuracy=0.5044799999999999

3 feature_encoder, relation_network = CNNEncoder1(), RelationNetwork1()
2020-10-25 11:56:13 load feature encoder success from ../models/fsl_sgd/2_400_64_5_1_fe_5way_1shot.pkl
2020-10-25 11:56:13 load relation network success from ../models/fsl_sgd/2_400_64_5_1_rn_5way_1shot.pkl
2020-10-25 11:57:55 Train 400 Accuracy: 0.7193333333333334
2020-10-25 11:57:55 Val   400 Accuracy: 0.5487777777777777
2020-10-25 12:02:00 episode=400, Mean Test accuracy=0.5206977777777778
"""


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.feature_encoder.eval()
    runner.relation_network.eval()
    runner.test_tool.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.feature_encoder.eval()
    runner.relation_network.eval()
    runner.test_tool.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
