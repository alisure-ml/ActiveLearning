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
from pn_miniimagenet_fsl_test_tool import TestTool
from pn_miniimagenet_ic_test_tool import ICTestTool
# from pn_miniimagenet_tool import Normalize, ProtoNet, RunnerTool
from pn_miniimagenet_tool import RunnerTool


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

        now_label_k_shot_index = self._get_samples_by_clustering_label(_now_label, True, num=self.num_shot)
        # now_label_k_shot_index = self._get_samples_by_clustering_label(_now_label, True, num=self.num_shot, now_index=now_index)

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
                if now_index in search_index_list:
                    search_index_list.remove(now_index)
                other_features = self.features[search_index_list]

                # sim_result = np.matmul(other_features, now_feature)
                now_features = np.tile(now_feature[None, ...], reps=[other_features.shape[0], 1])
                sim_result = np.sum(now_features * other_features, axis=-1)

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


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-16)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ProtoNet(nn.Module):

    def __init__(self, hid_dim, z_dim, has_norm=False):
        super().__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 41
        self.conv_block_2 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 21
        self.conv_block_3 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 10
        self.conv_block_4 = nn.Sequential(nn.Conv2d(hid_dim, z_dim, 3, padding=1),
                                          nn.BatchNorm2d(z_dim), nn.ReLU(), nn.MaxPool2d(2))  # 5

        self.has_norm = has_norm
        if self.has_norm:
            self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        if self.has_norm:
            out = out.view(out.shape[0], -1)
            out = self.l2norm(out)
        return out

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
        self.proto_net = RunnerTool.to_cuda(Config.proto_net)
        self.ic_model = RunnerTool.to_cuda(ICResNet(low_dim=Config.ic_out_dim))

        RunnerTool.to_cuda(self.proto_net.apply(RunnerTool.weights_init))
        RunnerTool.to_cuda(self.ic_model.apply(RunnerTool.weights_init))

        # optim
        self.proto_net_optim = torch.optim.SGD(
            self.proto_net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # loss
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())

        # Eval
        self.test_tool_fsl = TestTool(self.proto_test, data_root=Config.data_root,
                                      num_way=Config.num_way, num_shot=Config.num_shot,
                                      episode_size=Config.episode_size, test_episode=Config.test_episode,
                                      transform=self.task_train.transform_test)
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
                                       data_root=Config.data_root, batch_size=Config.batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim)
        pass

    def load_model(self):
        if Config.ic_pretrain_dir and os.path.exists(Config.ic_pretrain_dir):
            self.ic_model.load_state_dict(torch.load(Config.ic_pretrain_dir))
            Tools.print("load ic pretrain model success from {}".format(Config.ic_pretrain_dir))

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
        return log_p_y

    def proto_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        sample_z = self.proto_net(samples)  # 5x64*5*5
        batch_z = self.proto_net(batches)  # 75x64*5*5
        sample_z = sample_z.view(num_way, num_shot, -1)
        batch_z = batch_z.view(batch_num, -1)
        _, z_dim = batch_z.shape

        z_proto = sample_z.mean(1)
        z_proto_expand = z_proto.unsqueeze(0).expand(batch_num, num_way, z_dim)
        z_query_expand = batch_z.unsqueeze(1).expand(batch_num, num_way, z_dim)

        dists = torch.pow(z_query_expand - z_proto_expand, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def train(self):
        Tools.print()
        Tools.print("Training...")

        # Init Update
        if False:
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
            pass

        for epoch in range(1, 1 + Config.train_epoch):
            self.proto_net.train()
            self.ic_model.train()

            Tools.print()
            pn_lr= self.adjust_learning_rate(self.proto_net_optim, epoch,
                                             Config.first_epoch, Config.t_epoch, Config.learning_rate)
            ic_lr = self.adjust_learning_rate(self.ic_model_optim, epoch,
                                              Config.first_epoch, Config.t_epoch, Config.learning_rate)
            Tools.print('Epoch: [{}] pn_lr={} ic_lr={}'.format(epoch, pn_lr, ic_lr))

            self.produce_class.reset()
            Tools.print(self.task_train.classes)
            is_ok_total, is_ok_acc = 0, 0
            all_loss, all_loss_fsl, all_loss_ic = 0.0, 0.0, 0.0
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                log_p_y = self.proto(task_data)
                ic_out_logits, ic_out_l2norm = self.ic_model(task_data[:, -1])

                # 2
                ic_targets = self.produce_class.get_label(ic_labels)
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)

                # 3 loss
                loss_fsl = -(log_p_y * task_labels).sum() / task_labels.sum()
                loss_ic = self.ic_loss(ic_out_logits, ic_targets)
                loss = loss_fsl * Config.loss_fsl_ratio + loss_ic * Config.loss_ic_ratio
                all_loss += loss.item()
                all_loss_fsl += loss_fsl.item()
                all_loss_ic += loss_ic.item()

                # 4 backward
                if Config.train_ic:
                    self.ic_model.zero_grad()
                    loss_ic.backward()
                    self.ic_model_optim.step()
                    pass

                self.proto_net.zero_grad()
                loss_fsl.backward()
                self.proto_net_optim.step()

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
Norm=True
2020-11-29 20:55:31   2100 loss:1.623 fsl:0.604 ic:1.019 ok:0.256(9847/38400)
2020-11-29 20:55:31 Train: [2100] 8947/1754
2020-11-29 20:57:26 load feature encoder success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm_pn_5way_1shot.pkl
2020-11-29 20:57:26 load ic model success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm_ic_5way_1shot.pkl
2020-11-29 20:57:39 Epoch: 2100 Train 0.4941/0.7815 0.0000
2020-11-29 20:57:39 Epoch: 2100 Val   0.5780/0.9105 0.0000
2020-11-29 20:57:39 Epoch: 2100 Test  0.5575/0.9041 0.0000
2020-11-29 20:59:19 Train 2100 Accuracy: 0.4948888888888888
2020-11-29 20:59:19 Val   2100 Accuracy: 0.43855555555555553
2020-11-29 21:03:14 episode=2100, Mean Test accuracy=0.4551777777777778

1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_pn_5way_1shot.pkl
2020-12-01 06:16:29   2100 loss:1.669 fsl:0.660 ic:1.009 ok:0.264(10143/38400)
2020-12-01 06:16:29 Train: [2100] 8899/1741
2020-12-01 06:18:23 load feature encoder success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_pn_5way_1shot.pkl
2020-12-01 06:18:23 load ic model success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_ic_5way_1shot.pkl
2020-12-01 06:18:36 Epoch: 2100 Train 0.4982/0.7851 0.0000
2020-12-01 06:18:36 Epoch: 2100 Val   0.5806/0.9125 0.0000
2020-12-01 06:18:36 Epoch: 2100 Test  0.5617/0.9018 0.0000
2020-12-01 06:20:20 Train 2100 Accuracy: 0.4679999999999999
2020-12-01 06:20:20 Val   2100 Accuracy: 0.43411111111111117
2020-12-01 06:24:20 episode=2100, Mean Test accuracy=0.4441644444444445

2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/pn_final.pkl
2020-12-03 02:52:24   2100 loss:2.218 fsl:1.198 ic:1.020 ok:0.257(9879/38400)
2020-12-03 02:52:24 Train: [2100] 8943/1734
2020-12-03 02:54:19 load feature encoder success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/pn_final.pkl
2020-12-03 02:54:19 load ic model success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/ic_final.pkl
2020-12-03 02:54:19 Test 2100 .......
2020-12-03 02:54:33 Epoch: 2100 Train 0.4969/0.7808 0.0000
2020-12-03 02:54:33 Epoch: 2100 Val   0.5830/0.9116 0.0000
2020-12-03 02:54:33 Epoch: 2100 Test  0.5545/0.8988 0.0000
2020-12-03 02:56:17 Train 2100 Accuracy: 0.48133333333333334
2020-12-03 02:56:17 Val   2100 Accuracy: 0.4271111111111111
2020-12-03 03:00:33 episode=2100, Mean Test accuracy=0.41944

1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/pn_final.pkl
2020-12-03 22:35:51   2100 loss:2.223 fsl:1.195 ic:1.028 ok:0.258(9916/38400)
2020-12-03 22:35:51 Train: [2100] 9095/1777
2020-12-03 22:37:40 load feature encoder success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/pn_final.pkl
2020-12-03 22:37:40 load ic model success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/ic_final.pkl
2020-12-03 22:37:40 Test 2100 .......
2020-12-03 22:37:53 Epoch: 2100 Train 0.4945/0.7841 0.0000
2020-12-03 22:37:53 Epoch: 2100 Val   0.5787/0.9097 0.0000
2020-12-03 22:37:53 Epoch: 2100 Test  0.5623/0.8986 0.0000
2020-12-03 22:39:30 Train 2100 Accuracy: 0.4697777777777778
2020-12-03 22:39:30 Val   2100 Accuracy: 0.42266666666666663
2020-12-03 22:43:30 episode=2100, Mean Test accuracy=0.41988888888888887

2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/pn_final.pkl
2020-12-05 21:54:30 Train: [2100] 9139/1824
2020-12-05 21:56:52 load feature encoder success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/pn_final.pkl
2020-12-05 21:56:52 load ic model success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/ic_final.pkl
2020-12-05 21:56:52 Test 2100 .......
2020-12-05 21:57:06 Epoch: 2100 Train 0.4917/0.7822 0.0000
2020-12-05 21:57:06 Epoch: 2100 Val   0.5831/0.9143 0.0000
2020-12-05 21:57:06 Epoch: 2100 Test  0.5523/0.9018 0.0000
2020-12-05 21:58:58 Train 2100 Accuracy: 0.46344444444444444
2020-12-05 21:58:58 Val   2100 Accuracy: 0.41877777777777775
2020-12-05 22:03:15 episode=2100, Mean Test accuracy=0.41872444444444445

1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0/pn_final.pkl
2020-12-06 06:29:04   2100 loss:1.860 fsl:0.839 ic:1.021 ok:0.251(9624/38400)
2020-12-06 06:29:04 Train: [2100] 8988/1794
2020-12-06 06:30:59 load feature encoder success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0/pn_final.pkl
2020-12-06 06:30:59 load ic model success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0/ic_final.pkl
2020-12-06 06:30:59 Test 2100 .......
2020-12-06 06:31:11 Epoch: 2100 Train 0.4910/0.7782 0.0000
2020-12-06 06:31:11 Epoch: 2100 Val   0.5777/0.9125 0.0000
2020-12-06 06:31:11 Epoch: 2100 Test  0.5517/0.8986 0.0000
2020-12-06 06:32:53 Train 2100 Accuracy: 0.44122222222222224
2020-12-06 06:32:53 Val   2100 Accuracy: 0.4073333333333333
2020-12-06 06:36:58 episode=2100, Mean Test accuracy=0.4207466666666667

0_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/pn_final.pkl
2020-12-07 19:57:59   2100 loss:2.220 fsl:1.198 ic:1.022 ok:0.259(9942/38400)
2020-12-07 19:57:59 Train: [2100] 8995/1724
2020-12-07 19:59:50 load feature encoder success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/0_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/pn_final.pkl
2020-12-07 19:59:50 load ic model success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/0_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm/ic_final.pkl
2020-12-07 20:00:02 Epoch: 2100 Train 0.4942/0.7814 0.0000
2020-12-07 20:00:02 Epoch: 2100 Val   0.5776/0.9103 0.0000
2020-12-07 20:00:02 Epoch: 2100 Test  0.5520/0.9049 0.0000
2020-12-07 20:01:42 Train 2100 Accuracy: 0.465
2020-12-07 20:01:42 Val   2100 Accuracy: 0.4293333333333333
2020-12-07 20:05:43 episode=2100, Mean Test accuracy=0.41913333333333336

3_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_pn_5way_1shot.pkl
2020-12-09 19:01:29 load feature encoder success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/3_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_pn_5way_1shot.pkl
2020-12-09 19:01:29 load ic model success from ../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli/3_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_ic_5way_1shot.pkl
2020-12-09 19:01:29 Test 2100 .......
2020-12-09 19:01:52 Epoch: 2100 Train 0.4976/0.7848 0.0000
2020-12-09 19:01:52 Epoch: 2100 Val   0.5846/0.9125 0.0000
2020-12-09 19:01:52 Epoch: 2100 Test  0.5541/0.9012 0.0000
2020-12-09 19:03:57 Train 2100 Accuracy: 0.47322222222222216
2020-12-09 19:03:57 Val   2100 Accuracy: 0.41833333333333333
2020-12-09 19:10:56 episode=2100, Mean Test accuracy=0.42188444444444445

"""


class Config(object):
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 16

    num_way = 5
    num_shot = 1
    batch_size = 64

    val_freq = 10
    episode_size = 15
    test_episode = 600

    hid_dim = 64
    z_dim = 64

    proto_net = ProtoNet(hid_dim=hid_dim, z_dim=z_dim, has_norm=True)

    # ic
    ic_out_dim = 512
    ic_ratio = 1

    learning_rate = 0.01
    loss_fsl_ratio = 1.0
    loss_ic_ratio = 1.0

    train_epoch = 2100
    first_epoch, t_epoch = 500, 200
    adjust_learning_rate = RunnerTool.adjust_learning_rate1

    # train_ic = False
    train_ic = True
    ic_pretrain_dir = None
    if "Linux" in platform.platform() and not train_ic:
        _ic_pretrain_name = "2_2100_64_5_1_500_200_512_1_1.0_1.0_ic_5way_1shot.pkl"
        ic_pretrain_dir = "../models/two_ic_ufsl_2net_res_sgd_acc_duli/{}".format(_ic_pretrain_name)
        if not os.path.isdir(ic_pretrain_dir):
            ic_pretrain_dir = "../models/ic_res_no_val/1_32_512_1_500_200_0.01_ic.pkl"
        Tools.print(ic_pretrain_dir)
        pass

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        gpu_id, train_epoch, batch_size, num_way, num_shot, hid_dim, z_dim, first_epoch,
        t_epoch, ic_out_dim, ic_ratio, loss_fsl_ratio, loss_ic_ratio)
    Tools.print(model_name)

    _root_path = "../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli"
    pn_dir = Tools.new_dir("{}/{}_pn_{}way_{}shot.pkl".format(_root_path, model_name, num_way, num_shot))
    ic_dir = Tools.new_dir("{}/{}_ic_{}way_{}shot.pkl".format(_root_path, model_name, num_way, num_shot))

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    # runner.proto_net.eval()
    # runner.ic_model.eval()
    # runner.test_tool_ic.val(epoch=0, is_print=True)
    # runner.test_tool_fsl.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.proto_net.eval()
    runner.ic_model.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
