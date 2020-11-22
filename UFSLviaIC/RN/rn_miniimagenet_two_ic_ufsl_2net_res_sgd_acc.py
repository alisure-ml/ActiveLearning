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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from rn_miniimagenet_fsl_test_tool import TestTool
from rn_miniimagenet_ic_test_tool import ICTestTool
from rn_miniimagenet_tool import ICModel, Normalize, CNNEncoder, RelationNetwork, RunnerTool


##############################################################################################################


class MiniImageNetDataset(object):

    def __init__(self, data_list, num_way, num_shot):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot
        self.data_id = np.asarray(range(len(self.data_list)))

        self.classes = None
        self.features = None

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

    def set_samples_class(self, classes):
        self.classes = classes
        pass

    def set_samples_feature(self, features):
        self.features = features
        pass

    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, _, now_image_filename = now_label_image_tuple
        _now_label = self.classes[item]
        now_label_k_shot_index = self._get_samples_by_clustering_label(_now_label, True,
                                                                       num=self.num_shot, now_index=now_index)
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
            task_data.append(torch.unsqueeze(self.read_image(one[2], transform), dim=0))
            pass
        task_data = torch.cat(task_data)

        task_label = torch.Tensor([int(index in now_label_k_shot_index) for index in c_way_k_shot_index_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index, is_ok_list

    def _get_samples_by_clustering_label(self, label, is_same_label=False, num=1, now_index=None, k=1):
        if is_same_label:
            if now_index:
                now_feature = self.features[now_index]

                if k == 1:
                    search_index = self.data_id[self.classes == label]
                else:
                    top_k_class = np.argpartition(now_feature, -k)[-k:]
                    search_index = np.hstack([self.data_id[self.classes == one] for one in top_k_class])
                    pass

                search_index_list = list(search_index)
                search_index_list.remove(now_index)
                other_feature = self.features[search_index_list]
                sim_result = np.matmul(other_feature, now_feature)
                sort_result = np.argsort(sim_result)[::-1]
                return list(search_index[sort_result][0: num])
            return random.sample(list(np.squeeze(np.argwhere(self.classes == label), axis=1)), num)
        else:
            return random.sample(list(np.squeeze(np.argwhere(self.classes != label))), num)
        pass

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


class ICResNet(nn.Module):

    def __init__(self, low_dim=512):
        super().__init__()
        self.resnet = resnet18(num_classes=low_dim)
        self.l2norm = Normalize(2)
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
        self.data_train = MiniImageNetDataset.get_data_all(Config.data_root)
        self.task_train = MiniImageNetDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()
        self.task_train.set_samples_class(self.produce_class.classes)
        self.task_train.set_samples_feature(self.produce_class.features)

        # model
        self.feature_encoder = RunnerTool.to_cuda(Config.feature_encoder)
        self.relation_network = RunnerTool.to_cuda(Config.relation_network)
        self.ic_model = RunnerTool.to_cuda(ICResNet(low_dim=Config.ic_out_dim))

        RunnerTool.to_cuda(self.feature_encoder.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.relation_network.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.ic_model.apply(RunnerTool.weights_init))

        # optim
        self.feature_encoder_optim = torch.optim.SGD(
            self.feature_encoder.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.relation_network_optim = torch.optim.SGD(
            self.relation_network.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # loss
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())
        self.fsl_loss = RunnerTool.to_cuda(nn.MSELoss())

        # Eval
        self.test_tool_fsl = TestTool(self.compare_fsl_test, data_root=Config.data_root,
                                      num_way=Config.num_way, num_shot=Config.num_shot,
                                      episode_size=Config.episode_size, test_episode=Config.test_episode,
                                      transform=self.task_train.transform_test)
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
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

        # Init Update
        try:
            self.feature_encoder.eval()
            self.relation_network.eval()
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

        for epoch in range(Config.train_epoch):
            self.feature_encoder.train()
            self.relation_network.train()
            self.ic_model.train()

            Tools.print()
            fe_lr= self.adjust_learning_rate(self.feature_encoder_optim, epoch,
                                             Config.first_epoch, Config.t_epoch, Config.learning_rate)
            rn_lr = self.adjust_learning_rate(self.relation_network_optim, epoch,
                                              Config.first_epoch, Config.t_epoch, Config.learning_rate)
            ic_lr = self.adjust_learning_rate(self.ic_model_optim, epoch,
                                              Config.first_epoch, Config.t_epoch, Config.learning_rate)
            Tools.print('Epoch: [{}] fe_lr={} rn_lr={} ic_lr={}'.format(epoch, fe_lr, rn_lr, ic_lr))

            self.produce_class.reset()
            Tools.print(self.task_train.classes)
            is_ok_total, is_ok_acc = 0, 0
            all_loss, all_loss_fsl, all_loss_ic = 0.0, 0.0, 0.0
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                relations = self.compare_fsl(task_data)
                ic_out_logits, ic_out_l2norm = self.ic_model(task_data[:, -1])

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
                # torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
                # torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)
                # torch.nn.utils.clip_grad_norm_(self.ic_model.parameters(), 0.5)
                self.feature_encoder_optim.step()
                self.relation_network_optim.step()
                self.ic_model_optim.step()

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
                int(is_ok_acc) / int(is_ok_total), is_ok_acc, is_ok_total, ))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
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
1_2100_64_5_1_500_200_512_1_10.0_0.1_fe_5way_1shot.pkl
2020-10-31 22:11:06 load feature encoder success from ../models/two_ic_ufsl_2net_res_sgd/1_2100_64_5_1_500_200_512_1_10.0_0.1_fe_5way_1shot.pkl
2020-10-31 22:11:06 load relation network success from ../models/two_ic_ufsl_2net_res_sgd/1_2100_64_5_1_500_200_512_1_10.0_0.1_rn_5way_1shot.pkl
2020-10-31 22:11:06 load ic model success from ../models/two_ic_ufsl_2net_res_sgd/1_2100_64_5_1_500_200_512_1_10.0_0.1_ic_5way_1shot.pkl

2020-10-31 22:11:06 Test 2100 .......
2020-10-31 22:11:19 Epoch: 2100 Train 0.4134/0.7137 0.0000
2020-10-31 22:11:19 Epoch: 2100 Val   0.5223/0.8888 0.0000
2020-10-31 22:11:19 Epoch: 2100 Test  0.4773/0.8662 0.0000
2020-10-31 22:12:57 Train 2100 Accuracy: 0.4908888888888888
2020-10-31 22:12:57 Val   2100 Accuracy: 0.4351111111111111
2020-10-31 22:17:21 episode=2100, Mean Test accuracy=0.45217777777777785
"""


class Config(object):
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8

    num_way = 5
    num_shot = 1
    batch_size = 64

    val_freq = 10
    episode_size = 15
    test_episode = 600

    feature_encoder, relation_network = CNNEncoder(), RelationNetwork()

    # ic
    # ic_out_dim = 512
    ic_out_dim = 2560
    ic_ratio = 1

    learning_rate = 0.01
    loss_fsl_ratio = 10.0
    loss_ic_ratio = 0.1

    # train_epoch = 500
    # first_epoch, t_epoch = 300, 150
    # adjust_learning_rate = RunnerTool.adjust_learning_rate2

    train_epoch = 2100
    first_epoch, t_epoch = 500, 200
    adjust_learning_rate = RunnerTool.adjust_learning_rate1

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        gpu_id, train_epoch, batch_size, num_way, num_shot, first_epoch,
        t_epoch, ic_out_dim, ic_ratio, loss_fsl_ratio, loss_ic_ratio)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    _root_path = "../models/two_ic_ufsl_2net_res_sgd_acc"
    fe_dir = Tools.new_dir("{}/{}_fe_{}way_{}shot.pkl".format(_root_path, model_name, num_way, num_shot))
    rn_dir = Tools.new_dir("{}/{}_rn_{}way_{}shot.pkl".format(_root_path, model_name, num_way, num_shot))
    ic_dir = Tools.new_dir("{}/{}_ic_{}way_{}shot.pkl".format(_root_path, model_name, num_way, num_shot))
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    # runner.feature_encoder.eval()
    # runner.relation_network.eval()
    # runner.ic_model.eval()
    # runner.test_tool_ic.val(epoch=0, is_print=True)
    # runner.test_tool_fsl.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.feature_encoder.eval()
    runner.relation_network.eval()
    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
