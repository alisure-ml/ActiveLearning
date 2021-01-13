import os
import time
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
from mn_tool_ic_test import ICTestTool
from mn_tool_fsl_test import FSLTestTool
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34
from mn_tool_net import MatchingNet, Normalize, RunnerTool, ResNet12Small


##############################################################################################################


class CIFARFSDataset(object):

    def __init__(self, data_list, num_way, num_shot, image_size=32):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        self.data_id = np.asarray(range(len(self.data_list)))

        self.classes = None

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.transform_train_ic = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize
        ])

        if Config.aug_name == 1:
            normalize = transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
            self.transform_train_fsl = transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        elif Config.aug_name == 2:
            normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            self.transform_train_fsl = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            raise Exception(".................")

        self.transform_test =  transforms.Compose([transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

    def set_samples_class(self, classes):
        self.classes = classes
        pass

    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, _, now_image_filename = now_label_image_tuple
        _now_label = self.classes[item]

        now_label_k_shot_index = self._get_samples_by_clustering_label(_now_label, True, num=self.num_shot)
        is_ok_list = [self.data_list[one][1] == now_label_image_tuple[1] for one in now_label_k_shot_index]

        # 其他样本
        other_label_k_shot_index_list = self._get_samples_by_clustering_label(_now_label, False,
                                                                              num=self.num_shot * (self.num_way - 1))

        # c_way_k_shot
        c_way_k_shot_index_list = now_label_k_shot_index + other_label_k_shot_index_list
        random.shuffle(c_way_k_shot_index_list)

        if len(c_way_k_shot_index_list) != self.num_shot * self.num_way:
            return self.__getitem__(random.sample(list(range(0, len(self.data_list))), 1)[0])

        task_list = [self.data_list[index] for index in c_way_k_shot_index_list] + [now_label_image_tuple]

        task_data = []
        for one in task_list:
            transform = self.transform_train_ic if one[2] == now_image_filename else self.transform_train_fsl
            task_data.append(torch.unsqueeze(self.read_image(one, transform), dim=0))
            pass
        task_data = torch.cat(task_data)

        task_label = torch.Tensor([int(index in now_label_k_shot_index) for index in c_way_k_shot_index_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index, is_ok_list

    def _get_samples_by_clustering_label(self, label, is_same_label=False, num=1, now_index=None, k=1):
        if is_same_label:
            return random.sample(list(np.squeeze(np.argwhere(self.classes == label), axis=1)), num)
        else:
            return random.sample(list(np.squeeze(np.argwhere(self.classes != label))), num)
        pass

    def read_image(self, one, transform=None):
        image = Image.open(one[2]).convert('RGB')
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


class ICResNet(nn.Module):

    def __init__(self, resnet, low_dim=512, modify_head=False):
        super().__init__()
        self.resnet = resnet(num_classes=low_dim)
        self.l2norm = Normalize(2)
        if modify_head:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            pass
        pass

    def forward(self, x):
        out_logits = self.resnet(x)
        out_l2norm = self.l2norm(out_logits)
        return out_logits, out_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ProduceClass(object):

    def __init__(self, n_sample, out_dim, ratio=1.0):
        super().__init__()
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.class_per_num = self.n_sample // self.out_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = np.zeros(shape=(self.out_dim,), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample,), dtype=np.int)
        self.features = np.random.random(size=(self.n_sample, self.out_dim))
        pass

    def init(self):
        class_per_num = self.n_sample // self.out_dim
        self.class_num += class_per_num
        for i in range(self.out_dim):
            self.classes[i * class_per_num: (i + 1) * class_per_num] = i
            pass
        np.random.shuffle(self.classes)
        pass

    def reset(self):
        self.count = 0
        self.count_2 = 0
        self.class_num *= 0
        pass

    def cal_label(self, out, indexes):
        out_data = out.data.cpu()
        top_k = out_data.topk(self.out_dim, dim=1)[1].cpu()
        indexes_cpu = indexes.cpu()

        self.features[indexes_cpu] = out_data

        batch_size = top_k.size(0)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int)

        for i in range(batch_size):
            for j_index, j in enumerate(top_k[i]):
                if self.class_per_num > self.class_num[j]:
                    class_labels[i] = j
                    self.class_num[j] += 1
                    self.count += 1 if self.classes[indexes_cpu[i]] != j else 0
                    self.classes[indexes_cpu[i]] = j
                    self.count_2 += 1 if j_index != 0 else 0
                    break
                pass
            pass
        pass

    def get_label(self, indexes):
        _device = indexes.device
        return torch.tensor(self.classes[indexes.cpu()]).long().to(_device)

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0
        self.adjust_learning_rate = Config.adjust_learning_rate

        # all data
        self.data_train = CIFARFSDataset.get_data_all(Config.data_root)
        self.task_train = CIFARFSDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()
        self.task_train.set_samples_class(self.produce_class.classes)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        self.ic_model = RunnerTool.to_cuda(ICResNet(low_dim=Config.ic_out_dim,
                                                    resnet=Config.resnet, modify_head=Config.modify_head))
        self.norm = Normalize(2)
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.ic_model.apply(RunnerTool.weights_init))

        # optim
        self.matching_net_optim = torch.optim.SGD(
            self.matching_net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # loss
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())
        self.fsl_loss = RunnerTool.to_cuda(nn.MSELoss())

        # Eval
        self.test_tool_fsl = FSLTestTool(self.matching_test, data_root=Config.data_root,
                                         num_way=Config.num_way, num_shot=Config.num_shot,
                                         episode_size=Config.episode_size, test_episode=Config.test_episode,
                                         transform=self.task_train.transform_test)
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model, data_root=Config.data_root,
                                       transform=self.task_train.transform_test, batch_size=Config.batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim, k=Config.knn)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load matching net success from {}".format(Config.mn_dir))

        if os.path.exists(Config.ic_dir):
            self.ic_model.load_state_dict(torch.load(Config.ic_dir))
            Tools.print("load ic model success from {}".format(Config.ic_dir))
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

        # Init Update
        try:
            self.matching_net.eval()
            self.ic_model.eval()
            Tools.print("Init label {} .......")
            self.produce_class.reset()
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)
                ic_out_logits, ic_out_l2norm = self.ic_model(task_data[:, -1])
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2))
        finally:
            pass

        for epoch in range(1, 1 + Config.train_epoch):
            self.matching_net.train()
            self.ic_model.train()

            Tools.print()
            mn_lr= self.adjust_learning_rate(self.matching_net_optim, epoch,
                                             Config.first_epoch, Config.t_epoch, Config.learning_rate)
            ic_lr = self.adjust_learning_rate(self.ic_model_optim, epoch,
                                              Config.first_epoch, Config.t_epoch, Config.learning_rate)
            Tools.print('Epoch: [{}] mn_lr={} ic_lr={}'.format(epoch, mn_lr, ic_lr))

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
                loss = loss_fsl * Config.loss_fsl_ratio + loss_ic * Config.loss_ic_ratio
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
                epoch, all_loss / len(self.task_train_loader),
                all_loss_fsl / len(self.task_train_loader), all_loss_ic / len(self.task_train_loader),
                int(is_ok_acc) / int(is_ok_total), is_ok_acc, is_ok_total, ))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                self.matching_net.eval()
                self.ic_model.eval()

                self.test_tool_ic.val(epoch=epoch)
                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True, has_test=False)

                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.matching_net.state_dict(), Config.mn_dir)
                    torch.save(self.ic_model.state_dict(), Config.ic_dir)
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

    #######################################################################################
    num_workers = 16
    num_way = 5
    num_shot = 1
    val_freq = 10
    episode_size = 15
    test_episode = 600
    ic_out_dim = 1024
    ic_ratio = 1
    knn = 100
    learning_rate = 0.01
    loss_fsl_ratio = 1.0
    loss_ic_ratio = 1.0
    #######################################################################################

    #######################################################################################
    dataset_name = "CIFARFS"
    # dataset_name = "FC100"
    ic_out_dim = 512
    resnet, modify_head, ic_net_name = resnet34, True, "res34_head"
    matching_net, net_name, batch_size = MatchingNet(hid_dim=64, z_dim=64), "conv4", 64
    # matching_net, net_name, batch_size = ResNet12Small(avg_pool=True, drop_rate=0.1), "resnet12", 64
    ###############################################################################################

    ###############################################################################################
    if dataset_name == "CIFARFS":
        aug_name = 1  # other
        # aug_name = 2  # my
        train_epoch, first_epoch, t_epoch = 1600, 400, 200
        adjust_learning_rate = RunnerTool.adjust_learning_rate1
    else:
        aug_name = 1  # other
        # aug_name = 2  # my
        train_epoch, first_epoch, t_epoch = 1600, 400, 200
        adjust_learning_rate = RunnerTool.adjust_learning_rate1
        pass
    ###############################################################################################

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_aug{}".format(
        gpu_id, dataset_name, 32, net_name, train_epoch, batch_size, num_way, num_shot,
        first_epoch, t_epoch, ic_out_dim, ic_ratio, aug_name)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/{}'.format(dataset_name)
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/{}'.format(dataset_name)
    else:
        data_root = "F:\\data\\{}".format(dataset_name)

    _root_path = "../models_CIFARFS/mn/two_ic_ufsl_2net_res_sgd_acc_duli"
    mn_dir = Tools.new_dir("{}/{}_mn.pkl".format(_root_path, model_name))
    ic_dir = Tools.new_dir("{}/{}_ic.pkl".format(_root_path, model_name))

    Tools.print(model_name)
    Tools.print(data_root)
    pass


"""
0_CIFARFS_32_conv4_1600_64_5_1_400_200_1024_1_aug2_mn
2021-01-11 19:12:46 load matching net success from ../models_CIFARFS/mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_CIFARFS_32_conv4_1600_64_5_1_400_200_1024_1_aug2_mn.pkl
2021-01-11 19:12:46 load ic model success from ../models_CIFARFS/mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_CIFARFS_32_conv4_1600_64_5_1_400_200_1024_1_aug2_ic.pkl
2021-01-11 19:12:46 Test 1600 .......
2021-01-11 19:13:02 Epoch: 1600 Train 0.6052/0.8587 0.0000
2021-01-11 19:13:02 Epoch: 1600 Val   0.6386/0.9339 0.0000
2021-01-11 19:13:02 Epoch: 1600 Test  0.6769/0.9491 0.0000
2021-01-11 19:13:19 Train 1600 Accuracy: 0.5351111111111111
2021-01-11 19:13:36 Val   1600 Accuracy: 0.48088888888888887
2021-01-11 19:13:53 Test1 1600 Accuracy: 0.5091111111111112
2021-01-11 19:14:34 Test2 1600 Accuracy: 0.5064666666666666
2021-01-11 19:17:54 episode=1600, Test accuracy=0.5026
2021-01-11 19:17:54 episode=1600, Test accuracy=0.5054444444444445
2021-01-11 19:17:54 episode=1600, Test accuracy=0.5168222222222222
2021-01-11 19:17:54 episode=1600, Test accuracy=0.5128444444444445
2021-01-11 19:17:54 episode=1600, Test accuracy=0.5023777777777778
2021-01-11 19:17:54 episode=1600, Mean Test accuracy=0.5080177777777778

2021-01-12 22:33:23 load matching net success from ../models_CIFARFS/mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_CIFARFS_32_conv4_1600_64_5_1_400_200_1024_1_aug1_mn.pkl
2021-01-12 22:33:23 load ic model success from ../models_CIFARFS/mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_CIFARFS_32_conv4_1600_64_5_1_400_200_1024_1_aug1_ic.pkl
2021-01-12 22:33:23 Test 1600 .......
2021-01-12 22:33:38 Epoch: 1600 Train 0.6023/0.8591 0.0000
2021-01-12 22:33:38 Epoch: 1600 Val   0.6347/0.9325 0.0000
2021-01-12 22:33:38 Epoch: 1600 Test  0.6700/0.9464 0.0000
2021-01-12 22:33:53 Train 1600 Accuracy: 0.569
2021-01-12 22:34:07 Val   1600 Accuracy: 0.4922222222222223
2021-01-12 22:34:22 Test1 1600 Accuracy: 0.5224444444444445
2021-01-12 22:34:59 Test2 1600 Accuracy: 0.5124222222222222
2021-01-12 22:38:11 episode=1600, Test accuracy=0.5154666666666666
2021-01-12 22:38:11 episode=1600, Test accuracy=0.5166444444444445
2021-01-12 22:38:11 episode=1600, Test accuracy=0.5123777777777777
2021-01-12 22:38:11 episode=1600, Test accuracy=0.5114666666666666
2021-01-12 22:38:11 episode=1600, Test accuracy=0.5160222222222223
2021-01-12 22:38:11 episode=1600, Mean Test accuracy=0.5143955555555555
"""


"""
2021-01-12 15:54:44 load matching net success from ../models_CIFARFS/mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_FC100_32_conv4_1800_64_5_1_400_200_1024_1_1.0_1.0_aug1_mn.pkl
2021-01-12 15:54:44 load ic model success from ../models_CIFARFS/mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_FC100_32_conv4_1800_64_5_1_400_200_1024_1_1.0_1.0_aug1_ic.pkl
2021-01-12 15:54:44 Test 1800 .......
2021-01-12 15:54:59 Epoch: 1800 Train 0.6362/0.9015 0.0000
2021-01-12 15:54:59 Epoch: 1800 Val   0.4653/0.8093 0.0000
2021-01-12 15:54:59 Epoch: 1800 Test  0.4535/0.8300 0.0000
2021-01-12 15:55:13 Train 1800 Accuracy: 0.6184444444444444
2021-01-12 15:55:26 Val   1800 Accuracy: 0.3174444444444445
2021-01-12 15:55:40 Test1 1800 Accuracy: 0.33122222222222225
2021-01-12 15:56:16 Test2 1800 Accuracy: 0.3282
2021-01-12 15:59:18 episode=1800, Test accuracy=0.3421777777777778
2021-01-12 15:59:18 episode=1800, Test accuracy=0.3424888888888889
2021-01-12 15:59:18 episode=1800, Test accuracy=0.33497777777777776
2021-01-12 15:59:18 episode=1800, Test accuracy=0.3381555555555556
2021-01-12 15:59:18 episode=1800, Test accuracy=0.3277111111111111
2021-01-12 15:59:18 episode=1800, Mean Test accuracy=0.33710222222222225

2021-01-13 16:10:46 load matching net success from ../models_CIFARFS/mn/two_ic_ufsl_2net_res_sgd_acc_duli/3_FC100_32_resnet12_1600_64_5_1_400_200_1024_1_aug1_mn.pkl
2021-01-13 16:10:46 load ic model success from ../models_CIFARFS/mn/two_ic_ufsl_2net_res_sgd_acc_duli/3_FC100_32_resnet12_1600_64_5_1_400_200_1024_1_aug1_ic.pkl
2021-01-13 16:10:46 Test 1600 .......
2021-01-13 16:11:02 Epoch: 1600 Train 0.6388/0.9035 0.0000
2021-01-13 16:11:02 Epoch: 1600 Val   0.4647/0.8085 0.0000
2021-01-13 16:11:02 Epoch: 1600 Test  0.4537/0.8357 0.0000
2021-01-13 16:11:25 Train 1600 Accuracy: 0.6912222222222223
2021-01-13 16:11:48 Val   1600 Accuracy: 0.3488888888888889
2021-01-13 16:12:11 Test1 1600 Accuracy: 0.35244444444444445
2021-01-13 16:13:36 Test2 1600 Accuracy: 0.3566888888888889
2021-01-13 16:20:43 episode=1600, Test accuracy=0.3532666666666666
2021-01-13 16:20:43 episode=1600, Test accuracy=0.3622666666666666
2021-01-13 16:20:43 episode=1600, Test accuracy=0.3581333333333333
2021-01-13 16:20:43 episode=1600, Test accuracy=0.35364444444444443
2021-01-13 16:20:43 episode=1600, Test accuracy=0.34933333333333333
2021-01-13 16:20:43 episode=1600, Mean Test accuracy=0.35532888888888886
"""


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    # runner.matching_net.eval()
    # runner.ic_model.eval()
    # runner.test_tool_ic.val(epoch=0, is_print=True)
    # runner.test_tool_fsl.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.matching_net.eval()
    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
