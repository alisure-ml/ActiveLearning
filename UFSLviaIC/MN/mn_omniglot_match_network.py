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
from mn_tool_fsl_test import FSLTestTool
from mn_tool_net import Normalize, RunnerTool


##############################################################################################################


class OmniglotDataset(object):

    def __init__(self, data_list, num_way, num_shot):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass
        normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
        self.transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), normalize])

        # normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
        # self.transform = transforms.Compose([transforms.RandomRotation(30, fill=255),
        #                                      transforms.Resize(28),
        #                                      transforms.RandomCrop(28, padding=4, fill=255),
        #                                      transforms.ToTensor(), normalize])
        # self.transform_test = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), normalize])
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
        image = Image.open(image_path).convert("RGB")
        if transform is not None:
            image = transform(image)
        return image

    pass


##############################################################################################################


class MatchingNet(nn.Module):

    def __init__(self, hid_dim, z_dim):
        super().__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 41
        self.conv_block_2 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 21
        self.conv_block_3 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 10
        self.conv_block_4 = nn.Sequential(nn.Conv2d(hid_dim, z_dim, 3, padding=1),
                                          nn.BatchNorm2d(z_dim), nn.ReLU(), nn.MaxPool2d(2))  # 5
        pass

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = out.view(out.shape[0], -1)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class Runner(object):

    def __init__(self):
        # all data
        self.data_train = OmniglotDataset.get_data_all(Config.data_root)
        self.task_train = OmniglotDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # model
        self.matching_net = RunnerTool.to_cuda(MatchingNet(hid_dim=64, z_dim=64))
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))
        self.norm = Normalize(2)

        # loss
        self.loss = RunnerTool.to_cuda(nn.MSELoss())

        # optim
        self.matching_net_optim = torch.optim.SGD(
            self.matching_net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.test_tool = FSLTestTool(self.matching_test, data_root=Config.data_root,
                                     num_way=Config.num_way_test, num_shot=Config.num_shot,
                                     episode_size=Config.episode_size, test_episode=Config.test_episode,
                                     transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load proto net success from {}".format(Config.mn_dir))
        pass

    def matching(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.matching_net(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        # 特征
        z_support, z_query = z.split(Config.num_shot * Config.num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, Config.num_way * Config.num_shot, z_dim)
        z_query_expand = z_query.expand(z_batch_size, Config.num_way * Config.num_shot, z_dim)

        # 相似性
        z_support = self.norm(z_support)
        similarities = torch.sum(z_support * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(z_batch_size, Config.num_way, Config.num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def matching_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)  # 5x64*5*5
        batch_z = self.matching_net(batches)  # 75x64*5*5
        z_support = sample_z.view(Config.num_way_test * Config.num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, Config.num_way_test * Config.num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, Config.num_way_test * Config.num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, Config.num_way_test, Config.num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def train(self):
        Tools.print()
        Tools.print("Training...")

        best_accuracy = 0.0
        for epoch in range(1, 1 + Config.train_epoch):
            self.matching_net.train()

            Tools.print()
            mn_lr= Config.adjust_learning_rate(self.matching_net_optim, epoch,
                                               Config.first_epoch, Config.t_epoch, Config.learning_rate)
            Tools.print('Epoch: [{}] mn_lr={}'.format(epoch, mn_lr))

            all_loss = 0.0
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                # 1 calculate features
                predicts = self.matching(task_data)

                # 2 loss
                loss = self.loss(predicts, task_labels)
                all_loss += loss.item()

                # 3 backward
                self.matching_net.zero_grad()
                loss.backward()
                self.matching_net_optim.step()
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
                self.matching_net.eval()

                val_accuracy = self.test_tool.val(episode=epoch, is_print=True, has_test=False)
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.matching_net.state_dict(), Config.mn_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################


class Config(object):
    gpu_id = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    learning_rate = 0.01
    num_workers = 24

    val_freq = 5
    episode_size = 15
    test_episode = 600

    train_epoch = 200
    first_epoch, t_epoch = 100, 50
    adjust_learning_rate = RunnerTool.adjust_learning_rate2

    ###############################################################################################
    num_way = 10
    # num_way = 60
    num_shot = 1
    num_way_test = 5
    batch_size = 64
    ###############################################################################################

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(
        gpu_id, train_epoch, batch_size, num_way, num_way_test, num_shot, first_epoch, t_epoch)

    mn_dir = Tools.new_dir("../omniglot/models_mn/fsl_sgd_modify/{}.pkl".format(model_name))
    dataset_name = "omniglot_single"
    # dataset_name = "omniglot_rot"
    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/{}'.format(dataset_name)
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/{}'.format(dataset_name)
        if not os.path.isdir(data_root):
            data_root = '/home/ubuntu/Dataset/Partition1/ALISURE/Data/UFSL/{}'.format(dataset_name)
    else:
        data_root = "F:\\data\\{}".format(dataset_name)

    Tools.print(model_name)
    Tools.print(data_root)
    Tools.print(mn_dir)
    pass


##############################################################################################################


"""
2021-01-18 18:47:33 load proto net success from ../omniglot/models_mn/fsl_sgd_modify/0_400_64_10_5_1_200_100.pkl
2021-01-18 18:47:43 Train 400 Accuracy: 0.9930000000000001
2021-01-18 18:47:53 Val   400 Accuracy: 0.9546666666666666
2021-01-18 18:48:03 Test1 400 Accuracy: 0.9574444444444445
2021-01-18 18:48:42 Test2 400 Accuracy: 0.9558888888888888
2021-01-18 18:52:00 episode=400, Test accuracy=0.9545555555555556
2021-01-18 18:52:00 episode=400, Test accuracy=0.9567111111111111
2021-01-18 18:52:00 episode=400, Test accuracy=0.9577555555555555
2021-01-18 18:52:00 episode=400, Test accuracy=0.9564
2021-01-18 18:52:00 episode=400, Test accuracy=0.9582222222222223
2021-01-18 18:52:00 episode=400, Mean Test accuracy=0.956728888888889


2021-01-18 18:22:14 load proto net success from ../omniglot/models_mn/fsl_sgd_modify/0_200_64_10_5_1_100_50.pkl
2021-01-18 18:22:22 Train 200 Accuracy: 0.9976666666666667
2021-01-18 18:22:29 Val   200 Accuracy: 0.9714444444444447
2021-01-18 18:22:36 Test1 200 Accuracy: 0.9721111111111113
2021-01-18 18:23:04 Test2 200 Accuracy: 0.9715111111111111
2021-01-18 18:25:19 episode=200, Test accuracy=0.9686888888888889
2021-01-18 18:25:19 episode=200, Test accuracy=0.968488888888889
2021-01-18 18:25:19 episode=200, Test accuracy=0.9692888888888888
2021-01-18 18:25:19 episode=200, Test accuracy=0.9665777777777779
2021-01-18 18:25:19 episode=200, Test accuracy=0.9717999999999999
2021-01-18 18:25:19 episode=200, Mean Test accuracy=0.9689688888888888


2021-01-19 00:58:19 load proto net success from ../omniglot/models_mn/fsl_sgd_modify/0_200_64_10_5_1_100_50.pkl
2021-01-19 00:58:26 Train 200 Accuracy: 0.9945555555555555
2021-01-19 00:58:34 Val   200 Accuracy: 0.9847777777777778
2021-01-19 00:58:41 Test1 200 Accuracy: 0.9805555555555554
2021-01-19 00:59:12 Test2 200 Accuracy: 0.983688888888889
2021-01-19 01:01:48 episode=200, Test accuracy=0.9824444444444445
2021-01-19 01:01:48 episode=200, Test accuracy=0.9832444444444445
2021-01-19 01:01:48 episode=200, Test accuracy=0.9816222222222223
2021-01-19 01:01:48 episode=200, Test accuracy=0.9834666666666666
2021-01-19 01:01:48 episode=200, Test accuracy=0.9840444444444446
2021-01-19 01:01:48 episode=200, Mean Test accuracy=0.9829644444444444
"""


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    # runner.matching_net.eval()
    # runner.test_tool.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.matching_net.eval()
    runner.test_tool.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
