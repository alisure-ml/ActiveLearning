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
from torchvision.models import resnet18, resnet34
sys.path.append("../Common")
from UFSLTool import MyTransforms, MyDataset, TrainDataset, C4Net, Normalize, ProduceClass
from UFSLTool import RunnerTool, ResNet12Small, ICResNet, FSLTestTool, ICTestTool


##############################################################################################################


class Runner(object):

    def __init__(self, config):
        self.config = config

        # all data
        self.data_train = MyDataset.get_data_split(self.config.data_root, split=MyDataset.dataset_split_train)
        self.task_train = TrainDataset(
            self.data_train, self.config.num_way, self.config.num_shot, transform_train_ic=self.config.transform_train_ic,
            transform_train_fsl=self.config.transform_train_fsl, transform_test=self.config.transform_test)
        self.task_train_loader = DataLoader(self.task_train, self.config.batch_size, True, num_workers=self.config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), self.config.ic_out_dim, self.config.ic_ratio)
        self.produce_class.init()
        self.task_train.set_samples_class(self.produce_class.classes)

        # model
        self.norm = Normalize(2)
        self.matching_net = RunnerTool.to_cuda(self.config.matching_net)
        self.ic_model = RunnerTool.to_cuda(ICResNet(
            low_dim=self.config.ic_out_dim, resnet=self.config.resnet, modify_head=self.config.modify_head))
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.ic_model.apply(RunnerTool.weights_init))

        # optim
        self.matching_net_optim = torch.optim.SGD(
            self.matching_net.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # loss
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())
        self.fsl_loss = RunnerTool.to_cuda(nn.MSELoss())

        # Eval
        self.test_tool_fsl = FSLTestTool(self.matching_test, data_root=self.config.data_root,
                                      num_way=self.config.num_way, num_shot=self.config.num_shot,
                                      episode_size=self.config.episode_size, test_episode=self.config.test_episode,
                                      transform=self.task_train.transform_test, txt_path=self.config.log_file)
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
                                       data_root=self.config.data_root, batch_size=self.config.batch_size,
                                       num_workers=self.config.num_workers, ic_out_dim=self.config.ic_out_dim,
                                       transform=self.task_train.transform_test, txt_path=self.config.log_file)
        pass

    def load_model(self):
        mn_dir = self.config.mn_checkpoint if self.config.is_eval else self.config.mn_dir
        ic_dir = self.config.ic_checkpoint if self.config.is_eval else self.config.ic_dir
        if os.path.exists(mn_dir):
            self.matching_net.load_state_dict(torch.load(mn_dir))
            Tools.print("load matching net success from {}".format(mn_dir), txt_path=self.config.log_file)

        if os.path.exists(ic_dir):
            self.ic_model.load_state_dict(torch.load(ic_dir))
            Tools.print("load ic model success from {}".format(ic_dir), txt_path=self.config.log_file)
        pass

    def matching(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.matching_net(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        # 特征
        z_support, z_query = z.split(self.config.num_shot * self.config.num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, self.config.num_way * self.config.num_shot, z_dim)
        z_query_expand = z_query.expand(z_batch_size, self.config.num_way * self.config.num_shot, z_dim)

        # 相似性
        z_support = self.norm(z_support)
        similarities = torch.sum(z_support * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(z_batch_size, self.config.num_way, self.config.num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def matching_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)  # 5x64*5*5
        batch_z = self.matching_net(batches)  # 75x64*5*5
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
        best_accuracy = 0.0
        Tools.print("Training...", txt_path=self.config.log_file)

        # Init Update
        try:
            self.matching_net.eval()
            self.ic_model.eval()
            Tools.print("Init label {} .......", txt_path=self.config.log_file)
            self.produce_class.reset()
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)
                ic_out_logits, ic_out_l2norm = self.ic_model(task_data[:, -1])
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2), txt_path=self.config.log_file)
        finally:
            pass

        for epoch in range(1, 1 + self.config.train_epoch):
            self.matching_net.train()
            self.ic_model.train()

            Tools.print()
            mn_lr= self.config.adjust_learning_rate(self.matching_net_optim, epoch, self.config.first_epoch,
                                                    self.config.t_epoch, self.config.learning_rate)
            ic_lr = self.config.adjust_learning_rate(self.ic_model_optim, epoch, self.config.first_epoch,
                                                     self.config.t_epoch, self.config.learning_rate)
            Tools.print('Epoch: [{}] mn_lr={} ic_lr={}'.format(epoch, mn_lr, ic_lr), txt_path=self.config.log_file)

            self.produce_class.reset()
            Tools.print(self.task_train.classes)
            is_ok_total, is_ok_acc = 0, 0
            all_loss, all_loss_fsl, all_loss_ic = 0.0, 0.0, 0.0
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                relations = self.matching(task_data)
                ic_out_logits, ic_out_l2norm = self.ic_model(task_data[:, -1])

                # 2
                ic_targets = self.produce_class.get_label(ic_labels)
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)

                # 3 loss
                loss_fsl = self.fsl_loss(relations, task_labels)
                loss_ic = self.ic_loss(ic_out_logits, ic_targets)
                loss = loss_fsl * self.config.loss_fsl_ratio + loss_ic * self.config.loss_ic_ratio
                all_loss += loss.item()
                all_loss_fsl += loss_fsl.item()
                all_loss_ic += loss_ic.item()

                # 4 backward
                self.ic_model.zero_grad()
                loss_ic.backward()
                self.ic_model_optim.step()

                self.matching_net.zero_grad()
                loss_fsl.backward()
                self.matching_net_optim.step()

                # is ok
                is_ok_acc += torch.sum(torch.cat(task_ok))
                is_ok_total += torch.prod(torch.tensor(torch.cat(task_ok).shape))
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} fsl:{:.3f} ic:{:.3f} ok:{:.3f}({}/{})".format(
                epoch + 1, all_loss / len(self.task_train_loader),
                all_loss_fsl / len(self.task_train_loader), all_loss_ic / len(self.task_train_loader),
                int(is_ok_acc) / int(is_ok_total), is_ok_acc, is_ok_total), txt_path=self.config.log_file)
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2), txt_path=self.config.log_file)
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % self.config.val_freq == 0:
                self.matching_net.eval()
                self.ic_model.eval()

                self.test_tool_ic.val(epoch=epoch)
                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.matching_net.state_dict(), Tools.new_dir(self.config.mn_dir))
                    torch.save(self.ic_model.state_dict(), Tools.new_dir(self.config.ic_dir))
                    Tools.print("Save networks for epoch: {}".format(epoch), txt_path=self.config.log_file)
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################


class Config(object):

    def __init__(self, gpu_id=1, dataset_name=MyDataset.dataset_name_miniimagenet, is_conv_4=True, is_res34=True,
                 is_modify_head=True, is_eval=False, mn_checkpoint=None, ic_checkpoint=None):
        self.gpu_id = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.dataset_name = dataset_name
        self.is_conv_4 = is_conv_4
        self.is_res34 = is_res34
        self.modify_head = is_modify_head
        self.is_eval = is_eval

        self.num_workers = 8
        self.num_way = 5
        self.num_shot = 1
        self.val_freq = 10
        self.episode_size = 15
        self.test_episode = 600
        self.ic_out_dim = 512
        self.ic_ratio = 1
        self.learning_rate = 0.01
        self.loss_fsl_ratio = 1.0
        self.loss_ic_ratio = 1.0

        ###############################################################################################
        self.train_epoch = 1500
        self.first_epoch, self.t_epoch = 300, 200
        self.adjust_learning_rate = RunnerTool.adjust_learning_rate1
        ###############################################################################################

        ###############################################################################################
        self.is_png = True
        self.data_root = MyDataset.get_data_root(dataset_name=self.dataset_name, is_png=self.is_png)
        self.transform_train_ic, self.transform_train_fsl, self.transform_test = MyTransforms.get_transform(
            dataset_name=self.dataset_name, has_ic=True, is_fsl_simple=True, is_css=False)

        if self.is_res34:
            self.resnet = resnet34
            self.ic_net_name = "res34{}".format("_head" if self.modify_head else "")
        else:
            self.resnet = resnet18
            self.ic_net_name = "res18{}".format("_head" if self.modify_head else "")
            pass

        if self.is_conv_4:
            self.matching_net, self.batch_size, self.e_net_name = C4Net(hid_dim=64, z_dim=64), 64, "C4"
        else:
            self.matching_net, self.batch_size, self.e_net_name = ResNet12Small(avg_pool=True, drop_rate=0.1), 32, "R12S"
        ###############################################################################################

        self.model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}{}".format(
            self.gpu_id, self.ic_net_name, self.e_net_name, self.train_epoch, self.batch_size,
            self.num_way, self.num_shot, self.first_epoch, self.t_epoch, self.ic_out_dim,
            self.ic_ratio, self.loss_fsl_ratio, self.loss_ic_ratio, "_png" if self.is_png else "")

        self.time = Tools.get_format_time()
        _root_path = "../models_abl/{}/mn".format(self.dataset_name)
        self.mn_dir = "{}/{}_{}_mn.pkl".format(_root_path, self.time, self.model_name)
        self.ic_dir = "{}/{}_{}_ic.pkl".format(_root_path, self.time, self.model_name)
        self.log_file = self.ic_dir.replace(".pkl", ".txt")

        if self.is_eval:
            self.mn_checkpoint = mn_checkpoint
            self.ic_checkpoint = ic_checkpoint
            self.log_file = os.path.join(_root_path, "abl_{}_{}_{}.txt".format(
                self.time, self.ic_net_name, self.e_net_name))
            pass

        Tools.print(self.data_root, txt_path=self.log_file)
        Tools.print(self.model_name, txt_path=self.log_file)
        Tools.print(self.mn_dir, txt_path=self.log_file)
        Tools.print(self.ic_dir, txt_path=self.log_file)
        pass

    pass


##############################################################################################################


def train(gpu_id, dataset_name=MyDataset.dataset_name_miniimagenet, is_train=True, test_first=False):
    config = Config(gpu_id=gpu_id, dataset_name=dataset_name,
                    is_conv_4=True, is_res34=True, is_modify_head=True, is_eval=False)
    runner = Runner(config=config)

    if test_first:
        runner.matching_net.eval()
        runner.ic_model.eval()
        runner.test_tool_ic.val(epoch=0, is_print=True)
        runner.test_tool_fsl.val(episode=0, is_print=True, has_test=False)
        pass

    if is_train:
        runner.train()

    runner.load_model()
    runner.matching_net.eval()
    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=config.train_epoch, is_print=True)
    runner.test_tool_fsl.val(episode=config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=config.train_epoch, is_print=True)
    pass


def final_eval(gpu_id, mn_checkpoint, ic_checkpoint, dataset_name, is_conv_4, is_res34, is_modify_head, test_episode=1000):
    config = Config(gpu_id, dataset_name=dataset_name, is_conv_4=is_conv_4, is_res34=is_res34, is_modify_head=is_modify_head,
                    is_eval=True, mn_checkpoint=mn_checkpoint, ic_checkpoint=ic_checkpoint)
    runner = Runner(config=config)

    runner.load_model()
    runner.matching_net.eval()
    runner.ic_model.eval()

    ways, shots = MyDataset.get_ways_shots(dataset_name=dataset_name)
    for index, way in enumerate(ways):
        Tools.print("{}/{} way={}".format(index, len(ways), way))
        m, pm = runner.test_tool_fsl.eval(num_way=way, num_shot=1, episode_size=15, test_episode=test_episode)
        Tools.print("way={},shot=1,acc={},con={}".format(way, m, pm), txt_path=config.log_file)
    for index, shot in enumerate(shots):
        Tools.print("{}/{} shot={}".format(index, len(shots), shot))
        m, pm = runner.test_tool_fsl.eval(num_way=5, num_shot=shot, episode_size=15, test_episode=test_episode)
        Tools.print("way=5,shot={},acc={},con={}".format(shot, m, pm), txt_path=config.log_file)
    pass


##############################################################################################################


if __name__ == '__main__':
    # train(gpu_id=0, dataset_name=MyDataset.dataset_name_miniimagenet, is_train=True, test_first=False)
    # train(gpu_id=0, dataset_name=MyDataset.dataset_name_miniimagenet, is_train=False, test_first=False)
    gpu_id = 0
    checkpoint_path = "../models_abl/miniimagenet/mn"
    ic_checkpoint = os.path.join(checkpoint_path, "1_2100_64_5_1_500_200_512_1_1.0_1.0_ic.pkl")
    mn_checkpoint = os.path.join(checkpoint_path, "1_2100_64_5_1_500_200_512_1_1.0_1.0_mn.pkl")
    is_res34 = False
    is_modify_head = False
    is_conv_4 = True
    final_eval(gpu_id=gpu_id, mn_checkpoint=mn_checkpoint, ic_checkpoint=ic_checkpoint,
               dataset_name=MyDataset.dataset_name_miniimagenet,
               is_conv_4=is_conv_4, is_res34=is_res34, is_modify_head=is_modify_head)
    pass
