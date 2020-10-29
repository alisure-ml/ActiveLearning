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
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pn_miniimagenet_fsl_test_tool import TestTool
from pn_miniimagenet_ic_test_tool import ICTestTool
from pn_miniimagenet_tool import ProduceClass, ProtoNet, RunnerTool, ICProtoNet


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


class ProtoResNet(nn.Module):

    def __init__(self, low_dim):
        super().__init__()
        self.resnet18 = resnet18(num_classes=low_dim)
        pass

    def forward(self, x):
        out = self.resnet18(x)
        return out


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
        self.proto_net = RunnerTool.to_cuda(Config.proto_net)
        self.ic_model = RunnerTool.to_cuda(Config.ic_proto_net)

        RunnerTool.to_cuda(self.proto_net.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.ic_model.apply(RunnerTool.weights_init))

        # optim
        self.proto_net_optim = torch.optim.Adam(self.proto_net.parameters(), lr=Config.learning_rate)
        self.ic_model_optim = torch.optim.Adam(self.ic_model.parameters(), lr=Config.learning_rate)

        self.proto_net_scheduler = StepLR(self.proto_net_optim, Config.train_epoch // 3, gamma=0.5)
        self.ic_model_scheduler = StepLR(self.ic_model_optim, Config.train_epoch // 3, gamma=0.5)

        # loss
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())

        # Eval
        self.test_tool_fsl = TestTool(self.proto_test, data_root=Config.data_root,
                                      num_way=Config.num_way, num_shot=Config.num_shot,
                                      episode_size=Config.episode_size, test_episode=Config.test_episode,
                                      transform=self.task_train.transform_test)
        self.test_tool_ic = ICTestTool(feature_encoder=self.proto_net, ic_model=self.ic_model,
                                       data_root=Config.data_root, batch_size=Config.batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim)
        pass

    def load_model(self):
        if os.path.exists(Config.pn_dir):
            self.proto_net.load_state_dict(torch.load(Config.pn_dir))
            Tools.print("load feature encoder success from {}".format(Config.pn_dir))

        if os.path.exists(Config.ic_dir):
            self.ic_model.load_state_dict(torch.load(Config.ic_dir))
            Tools.print("load ic model success from {}".format(Config.ic_dir))
        pass

    def proto(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.proto_net(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        z_support, z_query = z.split(Config.num_shot * Config.num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, Config.num_way, Config.num_shot, z_dim)

        z_support_proto = z_support.mean(2)
        z_query_expand = z_query.expand(z_batch_size, Config.num_way, z_dim)

        dists = torch.pow(z_query_expand - z_support_proto, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y, z_query.squeeze(1)

    def proto_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.proto_net(samples)  # 5x64*5*5
        batch_z = self.proto_net(batches)  # 75x64*5*5
        sample_z = sample_z.view(Config.num_way, Config.num_shot, -1)
        batch_z = batch_z.view(batch_num, -1)
        _, z_dim = batch_z.shape

        z_proto = sample_z.mean(1)
        z_proto_expand = z_proto.unsqueeze(0).expand(batch_num, Config.num_way, z_dim)
        z_query_expand = batch_z.unsqueeze(1).expand(batch_num, Config.num_way, z_dim)

        dists = torch.pow(z_query_expand - z_proto_expand, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def train(self):
        Tools.print()
        Tools.print("Training...")

        # Init Update
        # try:
        #     if Config.has_eval:
        #         self.proto_net.eval()
        #         self.ic_model.eval()
        #     Tools.print("Init label {} .......")
        #     self.produce_class.reset()
        #     with torch.no_grad():
        #         for task_data, task_labels, task_index in tqdm(self.task_train_loader):
        #             ic_labels = RunnerTool.to_cuda(task_index[:, -1])
        #             task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)
        #             log_p_y, query_features = self.proto(task_data)
        #             ic_out_logits, ic_out_l2norm = self.ic_model(query_features)
        #             self.produce_class.cal_label(ic_out_l2norm, ic_labels)
        #             pass
        #         pass
        #     Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2))
        # finally:
        #     pass

        for epoch in range(Config.train_epoch):
            if Config.has_train:
                self.proto_net.train()
                self.ic_model.train()

            Tools.print()
            self.produce_class.reset()
            all_loss, all_loss_fsl, all_loss_ic = 0.0, 0.0, 0.0
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                log_p_y, query_features = self.proto(task_data)
                ic_out_logits, ic_out_l2norm = self.ic_model(query_features)

                # 2
                ic_targets = self.produce_class.get_label(ic_labels)
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)

                # 3 loss
                loss_fsl = -(log_p_y * task_labels).sum() / task_labels.sum() * Config.loss_fsl_ratio
                loss_ic = self.ic_loss(ic_out_logits, ic_targets) * Config.loss_ic_ratio
                loss = loss_fsl + loss_ic
                all_loss += loss.item()
                all_loss_fsl += loss_fsl.item()
                all_loss_ic += loss_ic.item()

                # 4 backward
                self.proto_net.zero_grad()
                self.ic_model.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.proto_net.parameters(), 0.5)
                # torch.nn.utils.clip_grad_norm_(self.ic_model.parameters(), 0.5)
                self.proto_net_optim.step()
                self.ic_model_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} fsl:{:.3f} ic:{:.3f} lr:{}".format(
                epoch + 1, all_loss / len(self.task_train_loader), all_loss_fsl / len(self.task_train_loader),
                all_loss_ic / len(self.task_train_loader), self.proto_net_scheduler.get_last_lr()))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            self.proto_net_scheduler.step()
            self.ic_model_scheduler.step()
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                if Config.has_eval:
                    self.proto_net.eval()
                    self.ic_model.eval()

                self.test_tool_ic.val(epoch=epoch)
                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True)

                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.proto_net.state_dict(), Config.pn_dir)
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
1_600_64_5_164_64_1600_512_1_10.0_0.1_True_True_ic_5way_1shot.pkl
2020-10-27 07:35:03 Test 600 .......
2020-10-27 07:35:14 Epoch: 600 Train 0.3724/0.7162 0.0000
2020-10-27 07:35:14 Epoch: 600 Val   0.4770/0.8821 0.0000
2020-10-27 07:35:14 Epoch: 600 Test  0.4375/0.8558 0.0000
2020-10-27 07:36:37 Train 600 Accuracy: 0.5754444444444444
2020-10-27 07:36:37 Val   600 Accuracy: 0.4755555555555556
2020-10-27 07:40:02 episode=600, Mean Test accuracy=0.46810222222222225
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

    z_dim = 512

    # ic
    ic_in_dim = z_dim  # 1600
    ic_out_dim = 512
    ic_ratio = 1

    proto_net = ProtoResNet(low_dim=z_dim)
    ic_proto_net = ICProtoNet(in_dim=ic_in_dim, out_dim=ic_out_dim)

    train_epoch = 600

    loss_fsl_ratio = 10.0
    loss_ic_ratio = 0.1

    has_train = True
    has_eval = True

    model_name = "2_{}_{}_{}_{}{}_{}_{}_{}_{}_{}_{}_{}".format(
        train_epoch, batch_size, num_way, num_shot, z_dim, ic_in_dim,
        ic_out_dim, ic_ratio, loss_fsl_ratio, loss_ic_ratio, has_train, has_eval)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pn_dir = Tools.new_dir("../models_pn/two_ic_fsl_large/{}_pn_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    ic_dir = Tools.new_dir("../models_pn/two_ic_fsl_large/{}_ic_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    # if Config.has_eval:
    #     runner.proto_net.eval()
    #     runner.ic_model.eval()
    # runner.test_tool_ic.val(epoch=0, is_print=True)
    # runner.test_tool_fsl.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    if Config.has_eval:
        runner.proto_net.eval()
        runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
