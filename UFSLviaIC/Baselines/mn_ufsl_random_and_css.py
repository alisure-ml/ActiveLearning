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

        self.transform_train, self.transform_test = transform_train, transform_test
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, _, now_image_filename = now_label_image_tuple

        # 其他样本
        index_list = list(range(len(self.data_list)))
        index_list.remove(now_index)
        c_way_k_shot_index_list = random.sample(index_list, self.num_shot * self.num_way)

        if not Config.is_random:
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
                                         num_way=Config.num_way, num_shot=Config.num_shot,
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

    def matching_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.net(samples)  # 5x64*5*5
        batch_z = self.net(batches)  # 75x64*5*5
        z_support = sample_z.view(Config.num_way * Config.num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, Config.num_way * Config.num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, Config.num_way * Config.num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, Config.num_way, Config.num_shot)
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
                    torch.save(self.net.state_dict(), Config.fe_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################


class Config(object):
    gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8

    num_way = 5
    num_shot = 1
    batch_size = 64

    val_freq = 10
    episode_size = 15
    test_episode = 600

    learning_rate = 0.01
    train_epoch = 1700
    first_epoch, t_epoch = 500, 200
    adjust_learning_rate = RunnerTool.adjust_learning_rate1

    ###############################################################################################
    is_random = True

    dataset_name = "miniimagenet"

    is_png = True
    # is_png = False

    net, net_name = C4Net(hid_dim=64, z_dim=64, has_norm=False), "conv4"
    # net, net_name = ResNet12Small(avg_pool=True, drop_rate=0.1), "res12"
    ###############################################################################################

    method_name = "random" if is_random else "css"
    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}{}".format(
        gpu_id, method_name, net_name, train_epoch, batch_size,
        num_way, num_shot, first_epoch, t_epoch, "_png" if is_png else "")
    net_dir = Tools.new_dir("../models_baseline/{}/{}.pkl".format(method_name, model_name))

    data_root = MyDataset.get_data_root(dataset_name=dataset_name, is_png=is_png)
    transform_train, transform_test = MyTransforms.get_transform(dataset_name=dataset_name,
                                                                 has_ic=False, is_fsl_simple=False)
    Tools.print(model_name)
    Tools.print(net_dir)
    Tools.print(data_root)
    pass


##############################################################################################################


if __name__ == '__main__':
    runner = Runner()

    runner.train()

    runner.load_model()
    runner.net.eval()
    runner.test_tool_fsl.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
