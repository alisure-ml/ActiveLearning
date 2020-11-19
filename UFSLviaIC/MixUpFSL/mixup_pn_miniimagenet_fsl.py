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
from torch.utils.data import DataLoader, Dataset
from mixup_pn_miniimagenet_fsl_test_tool import TestTool
from mixup_pn_miniimagenet_tool import ProtoNet, RunnerTool


##############################################################################################################


class MiniImageNetDataset(object):

    def __init__(self, data_list):
        self.data_list = data_list

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
        now_label_image_tuple_2 = random.sample(self.data_dict[now_label], 1)[0]

        # 其他样本
        other_label = list(self.data_dict.keys())
        other_label.remove(now_label)
        other_label = random.sample(other_label, 1)[0]
        other_label_image_tuple = random.sample(self.data_dict[other_label], 1)[0]

        task_tuple = [now_label_image_tuple, now_label_image_tuple_2, other_label_image_tuple]
        task_data = torch.cat([torch.unsqueeze(self.read_image(one[2], self.transform), dim=0) for one in task_tuple])
        task_index = torch.Tensor([one[0] for one in task_tuple]).long()
        return task_tuple, task_data, task_index

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    pass


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # all data
        self.data_train = MiniImageNetDataset.get_data_all(Config.data_root)
        self.task_train = MiniImageNetDataset(self.data_train)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, shuffle=True, num_workers=Config.num_workers)

        # model
        self.proto_net = RunnerTool.to_cuda(Config.proto_net)
        self.norm = RunnerTool.to_cuda(Normalize())
        RunnerTool.to_cuda(self.proto_net.apply(RunnerTool.weights_init))

        # optim
        self.proto_net_optim = torch.optim.Adam(self.proto_net.parameters(), lr=Config.learning_rate)
        self.proto_net_scheduler = StepLR(self.proto_net_optim, Config.train_epoch // 3, gamma=0.5)

        self.test_tool = TestTool(self.proto_test, data_root=Config.data_root,
                                  num_way=Config.num_way,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode,
                                  transform=self.task_train.transform_test)

        # loss
        self.triple_margin_loss = torch.nn.TripletMarginLoss()
        pass

    def load_model(self):
        if os.path.exists(Config.pn_dir):
            self.proto_net.load_state_dict(torch.load(Config.pn_dir))
            Tools.print("load proto net success from {}".format(Config.pn_dir))
        pass

    def mixup_loss(self, z, beta_lambda):
        batch_size, num, c, w, h = z.shape
        x_a, x_1, x_2, x_a1, x_12, x_a2 = z.split(split_size=1, dim=1)

        beta_lambda_tile = torch.tensor(np.tile(beta_lambda[..., None, None, None], [c, w, h]))
        beta_lambda_tile = RunnerTool.to_cuda(beta_lambda_tile)
        p_1_1, p_2_1, p_3_1, p_1_2, p_2_2, p_3_2 = beta_lambda_tile.split(split_size=1, dim=1)

        triple_1 = self.triple_margin_loss(x_1, x_a, x_2) + self.triple_margin_loss(x_a, x_1, x_2)
        triple_2 = self.triple_margin_loss(x_a1, x_a, x_2) + self.triple_margin_loss(x_a1, x_1, x_2)
        loss_triple = triple_1 + triple_2

        mixup_1 = torch.mean(torch.sum(torch.pow(p_1_1 * x_a + p_1_2 * x_1 - x_a1, 2), dim=1)) * Config.mix_ratio
        mixup_2 = torch.mean(torch.sum(torch.pow(p_2_1 * x_1 + p_2_2 * x_2 - x_12, 2), dim=1)) * Config.mix_ratio
        mixup_3 = torch.mean(torch.sum(torch.pow(p_3_1 * x_2 + p_3_2 * x_a - x_a2, 2), dim=1)) * Config.mix_ratio
        loss_mixup = mixup_1 + mixup_2 + mixup_3

        # 2 loss
        loss = loss_triple + loss_mixup

        return loss, loss_triple, loss_mixup

    def proto_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.proto_net(samples)  # 5x64*5*5
        batch_z = self.proto_net(batches)  # 75x64*5*5

        sample_z = self.norm(sample_z)
        batch_z = self.norm(batch_z)

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

        for epoch in range(Config.train_epoch):
            self.proto_net.train()

            Tools.print()
            all_loss, all_loss_triple, all_loss_mixup = 0.0, 0.0, 0.0
            for task_tuple, inputs, task_index in tqdm(self.task_train_loader):
                batch_size, num, c, w, h = inputs.shape
                beta = np.random.beta(1, 1, [batch_size, num])  # 64, 3
                beta_lambda = np.hstack([beta, 1 - beta])  # 64, 6
                beta_lambda_tile = np.tile(beta_lambda[..., None, None, None], [c, w, h])

                inputs_1 = torch.cat([inputs, inputs[:, 1:, ...], inputs[:, 0:1, ...]], dim=1) * beta_lambda_tile
                inputs_1 = (inputs_1[:, 0:num, ...] + inputs_1[:, num:, ...]).float()
                now_inputs = torch.cat([inputs, inputs_1], dim=1).view(-1, c, w, h)
                now_inputs = RunnerTool.to_cuda(now_inputs)

                # 1 calculate features
                net_out = self.proto_net(now_inputs)
                net_out = self.norm(net_out)

                _, out_c, out_w, out_h = net_out.shape
                z = net_out.view(batch_size, -1, out_c, out_w, out_h)

                # 2 calculate loss
                loss, loss_triple, loss_mixup = self.mixup_loss(z, beta_lambda=beta_lambda)
                all_loss += loss.item()
                all_loss_mixup += loss_mixup.item()
                all_loss_triple += loss_triple.item()

                # 3 backward
                self.proto_net.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.proto_net.parameters(), 0.5)
                self.proto_net_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} triple:{:.3f} mixup:{:.3f} lr:{}".format(
                epoch + 1, all_loss / len(self.task_train_loader), all_loss_triple / len(self.task_train_loader),
                all_loss_mixup / len(self.task_train_loader), self.proto_net_scheduler.get_last_lr()))

            self.proto_net_scheduler.step()
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                Tools.print()
                Tools.print("Test {} {} .......".format(epoch, Config.model_name))
                self.proto_net.eval()

                val_accuracy = self.test_tool.val(episode=epoch, is_print=True)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.proto_net.state_dict(), Config.pn_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


class Config(object):
    gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    train_epoch = 1000
    # train_epoch = 180
    learning_rate = 0.001
    num_workers = 8

    val_freq = 5

    num_way = 5
    num_shot = 1
    batch_size = 64

    episode_size = 15
    test_episode = 600

    mix_ratio = 100.0

    hid_dim = 64
    z_dim = 64

    proto_net = ProtoNet(hid_dim=hid_dim, z_dim=z_dim)

    model_name = "{}_{}_{}_{}_{}_{}_{}".format(gpu_id, train_epoch, val_freq, batch_size, hid_dim, z_dim, mix_ratio)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pn_dir = Tools.new_dir("../models_pn/mixup_fsl/{}_pn_{}way_{}shot.pkl".format(model_name, num_way, num_shot))

    Tools.print(model_name)
    pass


##############################################################################################################


"""

"""


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    # runner.proto_net.eval()
    # runner.test_tool.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.proto_net.eval()
    runner.test_tool.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
