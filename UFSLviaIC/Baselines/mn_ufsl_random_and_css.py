import os
import sys
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
sys.path.append("../Common")
from UFSLTool import MyTransforms, MyDataset, C4Net, Normalize, RunnerTool, ResNet12Small, FSLTestTool


##############################################################################################################


class RandomAndCssDataset(object):

    def __init__(self, data_list, num_way, num_shot, transform_train, transform_test):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        if Config.baseline_type == "cluster":
            cluster_data = Tools.read_from_pkl(Config.cluster_path)
            self.data_list, self.cluster_list = cluster_data["info"], cluster_data["cluster"]
            pass

        self.transform_train, self.transform_test = transform_train, transform_test
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if Config.baseline_type == "random" or Config.baseline_type == "css":
            return self._getitem_random_and_css(item)
        elif Config.baseline_type == "cluster":
            return self._getitem_cluster(item)
        else:
            raise Exception("..........")
        pass

    def _getitem_random_and_css(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, _, now_image_filename = now_label_image_tuple

        # 其他样本
        index_list = list(range(len(self.data_list)))
        index_list.remove(now_index)
        c_way_k_shot_index_list = random.sample(index_list, self.num_shot * self.num_way)

        if Config.baseline_type == "css":
            c_way_k_shot_index_list[0] = now_index
        label_index = c_way_k_shot_index_list[0]
        random.shuffle(c_way_k_shot_index_list)

        if len(c_way_k_shot_index_list) != self.num_shot * self.num_way:
            return self.__getitem__(random.sample(list(range(0, len(self.data_list))), 1)[0])

        #######################################################################################
        query_list = [now_label_image_tuple]
        support_list = [self.data_list[index] for index in c_way_k_shot_index_list]
        task_list = support_list + query_list

        support_data = [torch.unsqueeze(
            MyDataset.read_image(one[2], self.transform_train), dim=0) for one in support_list]
        query_data = [torch.unsqueeze(
            MyDataset.read_image(one[2], self.transform_train), dim=0) for one in query_list]
        task_data = torch.cat(support_data + query_data)
        #######################################################################################

        task_label = torch.Tensor([int(index == label_index) for index in c_way_k_shot_index_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index

    def _getitem_cluster(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, _, now_image_filename = now_label_image_tuple

        now_cluster = self.cluster_list[item]
        now_label_k_shot_index = random.sample(list(np.squeeze(np.argwhere(self.cluster_list == now_cluster),
                                                               axis=1)), self.num_shot)

        # 其他样本
        index_list = list(range(len(self.data_list)))
        index_list.remove(now_index)
        other_label_k_shot_index_list = random.sample(index_list, self.num_shot * (self.num_way - 1))

        c_way_k_shot_index_list = now_label_k_shot_index + other_label_k_shot_index_list
        random.shuffle(c_way_k_shot_index_list)

        if len(c_way_k_shot_index_list) != self.num_shot * self.num_way:
            return self.__getitem__(random.sample(list(range(0, len(self.data_list))), 1)[0])

        #######################################################################################
        query_list = [now_label_image_tuple]
        support_list = [self.data_list[index] for index in c_way_k_shot_index_list]
        task_list = support_list + query_list

        support_data = [torch.unsqueeze(
            MyDataset.read_image(one[2], self.transform_train), dim=0) for one in support_list]
        query_data = [torch.unsqueeze(
            MyDataset.read_image(one[2], self.transform_train), dim=0) for one in query_list]
        task_data = torch.cat(support_data + query_data)
        #######################################################################################

        task_label = torch.Tensor([int(index in now_label_k_shot_index) for index in c_way_k_shot_index_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        # all data
        self.data_train = MyDataset.get_data_split(Config.data_root, split="train")
        self.task_train = RandomAndCssDataset(self.data_train, Config.num_way, Config.num_shot,
                                              Config.transform_train, Config.transform_test)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # model
        self.net = RunnerTool.to_cuda(Config.net)
        RunnerTool.to_cuda(self.net.apply(RunnerTool.weights_init))
        self.norm = Normalize(2)

        # optim
        self.loss = RunnerTool.to_cuda(nn.MSELoss())
        self.net_optim = torch.optim.SGD(self.net.parameters(),
                                         lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # Eval
        self.test_tool_fsl = FSLTestTool(self.matching_test, data_root=Config.data_root,
                                         num_way=Config.num_way_test, num_shot=Config.num_shot,
                                         episode_size=Config.episode_size, test_episode=Config.test_episode,
                                         transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.net_dir):
            self.net.load_state_dict(torch.load(Config.net_dir))
            Tools.print("load net success from {}".format(Config.net_dir))
        pass

    def matching(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.net(data_x)
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

    def matching_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        sample_z = self.net(samples)  # 5x64*5*5
        batch_z = self.net(batches)  # 75x64*5*5
        z_support = sample_z.view(num_way * num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, num_way * num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, num_way * num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, num_way, num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def train(self):
        Tools.print()
        Tools.print("Training...")
        best_accuracy = 0.0

        for epoch in range(1, 1 + Config.train_epoch):
            self.net.train()

            Tools.print()
            all_loss = 0.0
            net_lr= Config.adjust_learning_rate(self.net_optim, epoch, Config.first_epoch,
                                                Config.t_epoch, Config.learning_rate)
            Tools.print('Epoch: [{}] net_lr={}'.format(epoch, net_lr))
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                relations = self.matching(task_data)

                # 2 loss
                loss = self.loss(relations, task_labels)
                all_loss += loss.item()

                # 3 backward
                self.net.zero_grad()
                loss.backward()
                self.net_optim.step()
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

                self.net.eval()

                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True)
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.net.state_dict(), Config.net_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################


class Config(object):
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 16

    num_way = 5
    num_way_test = 5
    num_shot = 1
    learning_rate = 0.01

    episode_size = 15
    test_episode = 600

    # batch_size = 64
    # train_epoch = 1700
    # first_epoch, t_epoch = 500, 200
    # adjust_learning_rate = RunnerTool.adjust_learning_rate1

    is_png = True

    ###############################################################################################
    baseline_type_list = ["random", "css", "cluster"]
    # baseline_type = "css"
    # baseline_type = "random"
    baseline_type = "cluster"
    ###############################################################################################

    ###############################################################################################
    # val_freq = 10
    # dataset_name = "miniimagenet"
    # train_epoch = 300
    # first_epoch, t_epoch = 100, 100
    # adjust_learning_rate = RunnerTool.adjust_learning_rate2
    # net, net_name, batch_size = C4Net(hid_dim=64, z_dim=64, has_norm=False), "conv4", 64

    # val_freq = 10
    # dataset_name = "miniimagenet"
    # train_epoch = 150
    # first_epoch, t_epoch = 80, 40
    # adjust_learning_rate = RunnerTool.adjust_learning_rate2
    # net, net_name, batch_size = ResNet12Small(avg_pool=True, drop_rate=0.1), "res12", 32
    ###############################################################################################

    ###############################################################################################
    # val_freq = 5
    # dataset_name = "tieredimagenet"
    # train_epoch = 150
    # first_epoch, t_epoch = 80, 40
    # adjust_learning_rate = RunnerTool.adjust_learning_rate2
    # net, net_name, batch_size = C4Net(hid_dim=64, z_dim=64, has_norm=False), "conv4", 64

    # val_freq = 2
    # dataset_name = "tieredimagenet"
    # # train_epoch = 50
    # # first_epoch, t_epoch = 30, 10
    # train_epoch = 30
    # first_epoch, t_epoch = 16, 8
    # adjust_learning_rate = RunnerTool.adjust_learning_rate2
    # net, net_name, batch_size = ResNet12Small(avg_pool=True, drop_rate=0.1), "res12", 32
    ###############################################################################################

    ###############################################################################################
    val_freq = 10
    num_way = 10
    num_way_test = 5
    dataset_name = "omniglot"
    train_epoch = 300
    first_epoch, t_epoch = 200, 50
    adjust_learning_rate = RunnerTool.adjust_learning_rate2
    net, net_name, batch_size = C4Net(hid_dim=64, z_dim=64, has_norm=False), "conv4", 64
    ###############################################################################################

    transform_train, transform_test = MyTransforms.get_transform(
        dataset_name=dataset_name, has_ic=False, is_fsl_simple=False, is_css=baseline_type=="css")

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}{}".format(
        gpu_id, baseline_type, net_name, train_epoch, batch_size,
        num_way, num_shot, first_epoch, t_epoch, "_png" if is_png else "")
    net_dir = Tools.new_dir("../models_baseline/{}/{}/{}.pkl".format(dataset_name, baseline_type, model_name))

    data_root = MyDataset.get_data_root(dataset_name=dataset_name, is_png=is_png)
    if baseline_type == "cluster":
        cluster_path = os.path.join("{}_feature".format(data_root), "train_cluster.pkl")

    Tools.print(model_name)
    Tools.print(net_dir)
    Tools.print(data_root)
    pass


# miniimagenet
"""
random
2020-12-23 19:47:58 load net success from ../models_baseline/random/2_random_conv4_300_64_5_1_100_100_png.pkl
2020-12-23 19:48:26 Train 300 Accuracy: 0.2741111111111111
2020-12-23 19:48:54 Val   300 Accuracy: 0.2573333333333333
2020-12-23 19:49:21 Test1 300 Accuracy: 0.2581111111111111
2020-12-23 19:50:40 Test2 300 Accuracy: 0.2480666666666667
2020-12-23 19:57:10 episode=300, Test accuracy=0.2539111111111111
2020-12-23 19:57:10 episode=300, Test accuracy=0.25624444444444444
2020-12-23 19:57:10 episode=300, Test accuracy=0.2527777777777778
2020-12-23 19:57:10 episode=300, Test accuracy=0.2500888888888889
2020-12-23 19:57:10 episode=300, Test accuracy=0.2459777777777778
2020-12-23 19:57:10 episode=300, Mean Test accuracy=0.2518

css
2020-12-23 17:09:50 load net success from ../models_baseline/css/2_css_conv4_300_64_5_1_100_100_png.pkl
2020-12-23 17:10:14 Train 300 Accuracy: 0.4298888888888889
2020-12-23 17:10:39 Val   300 Accuracy: 0.41122222222222227
2020-12-23 17:11:02 Test1 300 Accuracy: 0.40611111111111114
2020-12-23 17:12:16 Test2 300 Accuracy: 0.40897777777777783
2020-12-23 17:17:51 episode=300, Test accuracy=0.4
2020-12-23 17:17:51 episode=300, Test accuracy=0.41375555555555554
2020-12-23 17:17:51 episode=300, Test accuracy=0.40957777777777776
2020-12-23 17:17:51 episode=300, Test accuracy=0.40577777777777785
2020-12-23 17:17:51 episode=300, Test accuracy=0.4082222222222222
2020-12-23 17:17:51 episode=300, Mean Test accuracy=0.40746666666666664

cluster
2020-12-23 19:59:11 load net success from ../models_baseline/cluster/1_cluster_conv4_300_64_5_1_100_100_png.pkl
2020-12-23 19:59:33 Train 300 Accuracy: 0.42788888888888893
2020-12-23 19:59:55 Val   300 Accuracy: 0.39655555555555555
2020-12-23 20:00:17 Test1 300 Accuracy: 0.4052222222222222
2020-12-23 20:01:24 Test2 300 Accuracy: 0.39984444444444445
2020-12-23 20:06:24 episode=300, Test accuracy=0.40704444444444443
2020-12-23 20:06:24 episode=300, Test accuracy=0.4051777777777778
2020-12-23 20:06:24 episode=300, Test accuracy=0.4036
2020-12-23 20:06:24 episode=300, Test accuracy=0.40015555555555554
2020-12-23 20:06:24 episode=300, Test accuracy=0.40895555555555557
2020-12-23 20:06:24 episode=300, Mean Test accuracy=0.40498666666666666


css
2020-12-23 17:37:26 2_css_res12_300_32_5_1_100_100_png
2020-12-23 17:37:26 ../models_baseline/css/2_css_res12_300_32_5_1_100_100_png.pkl
2020-12-23 17:37:26 /mnt/4T/Data/data/miniImagenet/miniImageNet_png
2020-12-24 09:19:46 Test 130 2_css_res12_300_32_5_1_100_100_png .......
2020-12-24 09:20:14 Train 130 Accuracy: 0.48033333333333333
2020-12-24 09:20:41 Val   130 Accuracy: 0.44133333333333324
2020-12-24 09:21:09 Test1 130 Accuracy: 0.4501111111111112
2020-12-24 09:22:41 Test2 130 Accuracy: 0.45168888888888886

2020-12-25 00:11:03 load net success from ../models_baseline/css/2_css_res12_300_32_5_1_100_100_png.pkl
2020-12-25 00:11:40 Train 300 Accuracy: 0.46688888888888885
2020-12-25 00:12:17 Val   300 Accuracy: 0.4453333333333333
2020-12-25 00:12:53 Test1 300 Accuracy: 0.44400000000000006
2020-12-25 00:15:19 Test2 300 Accuracy: 0.4496
2020-12-25 00:27:26 episode=300, Test accuracy=0.44580000000000003
2020-12-25 00:27:26 episode=300, Test accuracy=0.4488444444444444
2020-12-25 00:27:26 episode=300, Test accuracy=0.4464222222222223
2020-12-25 00:27:26 episode=300, Test accuracy=0.4468222222222223
2020-12-25 00:27:26 episode=300, Test accuracy=0.4527777777777777
2020-12-25 00:27:26 episode=300, Mean Test accuracy=0.4481333333333334


res12 random
2021-01-02 17:00:53 ../models_baseline/random/1_random_res12_300_32_5_1_100_100_png.pkl
2021-01-02 17:00:53 /mnt/4T/Data/data/miniImagenet/miniImageNet_png
2021-01-02 17:00:56 load net success from ../models_baseline/random/1_random_res12_300_32_5_1_100_100_png.pkl
2021-01-02 17:01:23 Train 300 Accuracy: 0.30333333333333334
2021-01-02 17:01:50 Val   300 Accuracy: 0.28477777777777774
2021-01-02 17:02:17 Test1 300 Accuracy: 0.303
2021-01-02 17:03:49 Test2 300 Accuracy: 0.3009555555555556
2021-01-02 17:11:28 episode=300, Test accuracy=0.29944444444444446
2021-01-02 17:11:28 episode=300, Test accuracy=0.29713333333333336
2021-01-02 17:11:28 episode=300, Test accuracy=0.29606666666666664
2021-01-02 17:11:28 episode=300, Test accuracy=0.3001111111111111
2021-01-02 17:11:28 episode=300, Test accuracy=0.2969555555555556
2021-01-02 17:11:28 episode=300, Mean Test accuracy=0.2979422222222223

res12 cluster
2021-01-03 13:43:05 load net success from ../models_baseline/cluster/2_cluster_res12_300_32_5_1_100_100_png.pkl
2021-01-03 13:43:33 Train 300 Accuracy: 0.5569999999999999
2021-01-03 13:44:01 Val   300 Accuracy: 0.46144444444444443
2021-01-03 13:44:29 Test1 300 Accuracy: 0.4706666666666666
2021-01-03 13:46:04 Test2 300 Accuracy: 0.47948888888888885
2021-01-03 13:54:02 episode=300, Test accuracy=0.4766444444444445
2021-01-03 13:54:02 episode=300, Test accuracy=0.4850444444444445
2021-01-03 13:54:02 episode=300, Test accuracy=0.48766666666666664
2021-01-03 13:54:02 episode=300, Test accuracy=0.48335555555555554
2021-01-03 13:54:02 episode=300, Test accuracy=0.4793777777777778
2021-01-03 13:54:02 episode=300, Mean Test accuracy=0.4824177777777778
"""


# tieredimagenet
"""
2021-01-05 02:49:08 load net success from ../models_baseline/css/1_css_conv4_150_64_5_1_80_120_png.pkl
2021-01-05 02:49:37 Train 150 Accuracy: 0.39644444444444443
2021-01-05 02:50:05 Val   150 Accuracy: 0.40644444444444444
2021-01-05 02:50:33 Test1 150 Accuracy: 0.4
2021-01-05 02:51:43 Test2 150 Accuracy: 0.4020222222222222
2021-01-05 02:57:29 episode=150, Test accuracy=0.40624444444444446
2021-01-05 02:57:29 episode=150, Test accuracy=0.40786666666666666
2021-01-05 02:57:29 episode=150, Test accuracy=0.40915555555555555
2021-01-05 02:57:29 episode=150, Test accuracy=0.4118222222222222
2021-01-05 02:57:29 episode=150, Test accuracy=0.40286666666666665
2021-01-05 02:57:29 episode=150, Mean Test accuracy=0.4075911111111111

2021-01-06 19:05:28 load net success from ../models_baseline/cluster/tieredimagenet/2_cluster_conv4_150_64_5_1_80_120_png.pkl
2021-01-06 19:05:56 Train 150 Accuracy: 0.43644444444444447
2021-01-06 19:06:24 Val   150 Accuracy: 0.41111111111111115
2021-01-06 19:06:52 Test1 150 Accuracy: 0.4228888888888889
2021-01-06 19:08:01 Test2 150 Accuracy: 0.4120666666666667
2021-01-06 19:13:50 episode=150, Test accuracy=0.41531111111111113
2021-01-06 19:13:50 episode=150, Test accuracy=0.4204222222222222
2021-01-06 19:13:50 episode=150, Test accuracy=0.4246888888888889
2021-01-06 19:13:50 episode=150, Test accuracy=0.42717777777777777
2021-01-06 19:13:50 episode=150, Test accuracy=0.42033333333333334
2021-01-06 19:13:50 episode=150, Mean Test accuracy=0.4215866666666666


2021-01-08 07:53:08 load net success from ../models_baseline/tieredimagenet/css/1_css_res12_50_32_5_1_30_40_png.pkl
2021-01-08 07:53:43 Train 50 Accuracy: 0.4575555555555556
2021-01-08 07:54:16 Val   50 Accuracy: 0.44933333333333336
2021-01-08 07:54:49 Test1 50 Accuracy: 0.43322222222222223
2021-01-08 07:56:27 Test2 50 Accuracy: 0.4276222222222222
2021-01-08 08:04:34 episode=50, Test accuracy=0.4270444444444445
2021-01-08 08:04:34 episode=50, Test accuracy=0.42840000000000006
2021-01-08 08:04:34 episode=50, Test accuracy=0.4403333333333334
2021-01-08 08:04:34 episode=50, Test accuracy=0.42684444444444447
2021-01-08 08:04:34 episode=50, Test accuracy=0.42875555555555556
2021-01-08 08:04:34 episode=50, Mean Test accuracy=0.4302755555555556

2021-01-08 12:56:24 load net success from ../models_baseline/tieredimagenet/cluster/2_cluster_res12_30_32_5_1_16_24_png.pkl
2021-01-08 12:56:59 Train 30 Accuracy: 0.44433333333333336
2021-01-08 12:57:34 Val   30 Accuracy: 0.43855555555555553
2021-01-08 12:58:08 Test1 30 Accuracy: 0.4373333333333333
2021-01-08 12:59:47 Test2 30 Accuracy: 0.4354888888888889
2021-01-08 13:08:12 episode=30, Test accuracy=0.4399333333333334
2021-01-08 13:08:12 episode=30, Test accuracy=0.44544444444444453
2021-01-08 13:08:12 episode=30, Test accuracy=0.4553333333333333
2021-01-08 13:08:12 episode=30, Test accuracy=0.4556888888888889
2021-01-08 13:08:12 episode=30, Test accuracy=0.44133333333333336
2021-01-08 13:08:12 episode=30, Mean Test accuracy=0.44754666666666676
"""


# omniglot
"""
2021-01-20 12:24:32 load net success from ../models_baseline/omniglot/random/1_random_conv4_300_64_10_1_200_50_png.pkl
2021-01-20 12:24:41 Train 300 Accuracy: 0.6255555555555556
2021-01-20 12:24:52 Val   300 Accuracy: 0.5641111111111111
2021-01-20 12:25:03 Test1 300 Accuracy: 0.5817777777777777
2021-01-20 12:25:56 Test2 300 Accuracy: 0.5760222222222222
2021-01-20 12:29:48 episode=300, Test accuracy=0.5779777777777777
2021-01-20 12:29:48 episode=300, Test accuracy=0.5778
2021-01-20 12:29:48 episode=300, Test accuracy=0.5824222222222223
2021-01-20 12:29:48 episode=300, Test accuracy=0.5828
2021-01-20 12:29:48 episode=300, Test accuracy=0.5864
2021-01-20 12:29:48 episode=300, Mean Test accuracy=0.58148

2021-01-20 13:53:27 load net success from ../models_baseline/omniglot/css/2_css_conv4_300_64_10_1_200_50_png.pkl
2021-01-20 13:53:35 Train 300 Accuracy: 0.8953333333333334
2021-01-20 13:53:43 Val   300 Accuracy: 0.838
2021-01-20 13:53:51 Test1 300 Accuracy: 0.8336666666666668
2021-01-20 13:54:30 Test2 300 Accuracy: 0.8367555555555557
2021-01-20 13:57:48 episode=300, Test accuracy=0.8388888888888889
2021-01-20 13:57:48 episode=300, Test accuracy=0.8383333333333334
2021-01-20 13:57:48 episode=300, Test accuracy=0.838
2021-01-20 13:57:48 episode=300, Test accuracy=0.8362888888888889
2021-01-20 13:57:48 episode=300, Test accuracy=0.842488888888889
2021-01-20 13:57:48 episode=300, Mean Test accuracy=0.8388

2021-01-20 14:11:56 load net success from ../models_baseline/omniglot/cluster/3_cluster_conv4_300_64_10_1_200_50_png.pkl
2021-01-20 14:12:04 Train 300 Accuracy: 0.9062222222222223
2021-01-20 14:12:11 Val   300 Accuracy: 0.851111111111111
2021-01-20 14:12:18 Test1 300 Accuracy: 0.8521111111111112
2021-01-20 14:12:48 Test2 300 Accuracy: 0.8570666666666665
2021-01-20 14:15:33 episode=300, Test accuracy=0.8579777777777777
2021-01-20 14:15:33 episode=300, Test accuracy=0.8583555555555555
2021-01-20 14:15:33 episode=300, Test accuracy=0.8548666666666668
2021-01-20 14:15:33 episode=300, Test accuracy=0.8580222222222221
2021-01-20 14:15:33 episode=300, Test accuracy=0.8612444444444445
2021-01-20 14:15:33 episode=300, Mean Test accuracy=0.8580933333333334
"""

##############################################################################################################


if __name__ == '__main__':
    runner = Runner()

    runner.train()

    runner.load_model()
    runner.net.eval()
    runner.test_tool_fsl.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
