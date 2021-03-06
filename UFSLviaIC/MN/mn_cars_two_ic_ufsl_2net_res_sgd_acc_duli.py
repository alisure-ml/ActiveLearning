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
from mn_tool_ic_test import ICTestTool
from mn_tool_fsl_test import FSLTestTool
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34
from mn_tool_net import MatchingNet, Normalize, RunnerTool, ResNet12Small


##############################################################################################################


class CarsDataset(object):

    def __init__(self, data_list, num_way, num_shot, image_size=84):
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

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.transform_train_ic = transforms.Compose([
        #     transforms.Resize([int(image_size * 1.25), int(image_size * 1.25)]),
        #     transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(0.2),
        #     transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        # self.transform_train_fsl = transforms.Compose([
        #     transforms.Resize([int(image_size * 1.25), int(image_size * 1.25)]),
        #     transforms.RandomCrop(image_size),
        #     transforms.ColorJitter(0.4, 0.4, 0.4), transforms.RandomGrayscale(0.2),
        #     transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        # self.transform_test = transforms.Compose([transforms.Resize([int(image_size * 1.25), int(image_size * 1.25)]),
        #                                           transforms.CenterCrop(image_size), transforms.ToTensor(), normalize])
        # self.transform_train_ic = transforms.Compose([
        #     transforms.Resize([int(image_size * 1.50), int(image_size * 1.50)]),
        #     transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(0.2),
        #     transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        # self.transform_train_fsl = transforms.Compose([
        #     transforms.Resize([int(image_size * 1.50), int(image_size * 1.50)]),
        #     transforms.RandomCrop(image_size),
        #     transforms.ColorJitter(0.4, 0.4, 0.4), transforms.RandomGrayscale(0.2),
        #     transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        # self.transform_test = transforms.Compose([transforms.Resize([int(image_size * 1.50), int(image_size * 1.50)]),
        #                                           transforms.CenterCrop(image_size), transforms.ToTensor(), normalize])
        self.transform_train_ic = transforms.Compose([
            transforms.Resize([int(image_size * 1.50), int(image_size * 1.50)]),
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_train_fsl = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.Resize([int(image_size * 1.50), int(image_size * 1.50)]),
            transforms.RandomCrop(image_size),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), transforms.RandomGrayscale(0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.Resize([int(image_size * 1.50), int(image_size * 1.50)]),
                                                  transforms.CenterCrop(image_size), transforms.ToTensor(), normalize])
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
        image_list = os.listdir(train_folder)
        for label in image_list:
            now_class_path = os.path.join(train_folder, label)
            if os.path.isdir(now_class_path):
                for time in range(Config.ic_times):
                    for name in os.listdir(now_class_path):
                        data_train_list.append((count_class, os.path.join(now_class_path, name)))
                        pass
                    pass
                count_class += 1
            pass

        np.random.shuffle(data_train_list)
        data_train_list_final = []
        for index, data_train in enumerate(data_train_list):
            data_train_list_final.append((index, data_train[0], data_train[1]))
            pass

        return data_train_list_final

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
        self.data_train = CarsDataset.get_data_all(Config.data_root)
        self.task_train = CarsDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()
        self.task_train.set_samples_class(self.produce_class.classes)
        self.task_train.set_samples_feature(self.produce_class.features)

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
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 16

    num_way = 5
    num_shot = 1
    # batch_size = 64

    # val_freq = 20
    val_freq = 10
    episode_size = 15
    test_episode = 600

    ic_ratio = 1
    knn = 50

    learning_rate = 0.01
    loss_fsl_ratio = 1.0
    loss_ic_ratio = 1.0

    ###############################################################################################
    # resnet = resnet18
    resnet = resnet34

    # modify_head = False
    modify_head = True

    # matching_net, net_name, batch_size = MatchingNet(hid_dim=64, z_dim=64), "conv4", 64
    matching_net, net_name, batch_size = ResNet12Small(avg_pool=True, drop_rate=0.1), "resnet12", 32

    # ic_times = 2
    # ic_out_dim = 512

    ic_out_dim = 2048
    ic_times = 5

    train_epoch = 1200
    first_epoch, t_epoch = 400, 200
    adjust_learning_rate = RunnerTool.adjust_learning_rate1

    # class_split = "256_png"
    class_split = "256_png_7"
    ###############################################################################################

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}{}_rot".format(
        gpu_id, class_split, net_name, train_epoch, batch_size, num_way, num_shot, first_epoch, t_epoch,
        ic_out_dim, ic_ratio, loss_fsl_ratio, loss_ic_ratio, ic_times, "_head" if modify_head else "")

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/Cars'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/Cars'
    else:
        data_root = "F:\\data\\Cars"
    data_root = os.path.join(data_root, class_split)

    _root_path = "../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli"
    mn_dir = Tools.new_dir("{}/{}_mn.pkl".format(_root_path, model_name))
    ic_dir = Tools.new_dir("{}/{}_ic.pkl".format(_root_path, model_name))

    Tools.print(model_name)
    Tools.print(data_root)
    pass


"""
2020-12-26 08:31:00 load matching net success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/2_256_png_7_conv4_1200_64_5_1_400_200_512_1_1.0_1.0_2_mn.pkl
2020-12-26 08:31:00 load ic model success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/2_256_png_7_conv4_1200_64_5_1_400_200_512_1_1.0_1.0_2_ic.pkl
2020-12-26 08:31:00 Test 1200 .......
2020-12-26 08:31:06 Epoch: 1200 Train 0.1701/0.3742 0.0000
2020-12-26 08:31:06 Epoch: 1200 Val   0.2972/0.6593 0.0000
2020-12-26 08:31:06 Epoch: 1200 Test  0.2154/0.5232 0.0000
2020-12-26 08:31:30 Train 1200 Accuracy: 0.3298888888888889
2020-12-26 08:31:54 Val   1200 Accuracy: 0.29388888888888887
2020-12-26 08:32:20 Test1 1200 Accuracy: 0.3442222222222222
2020-12-26 08:34:05 Test2 1200 Accuracy: 0.3370444444444445
2020-12-26 08:42:24 episode=1200, Test accuracy=0.33308888888888893
2020-12-26 08:42:24 episode=1200, Test accuracy=0.3329555555555555
2020-12-26 08:42:24 episode=1200, Test accuracy=0.3326
2020-12-26 08:42:24 episode=1200, Test accuracy=0.3360222222222222
2020-12-26 08:42:24 episode=1200, Test accuracy=0.33686666666666665
2020-12-26 08:42:24 episode=1200, Mean Test accuracy=0.33430666666666664


2020-12-27 13:17:45 load matching net success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/1_256_png_7_conv4_1200_64_5_1_400_200_512_1_1.0_1.0_2_head_mn.pkl
2020-12-27 13:17:46 load ic model success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/1_256_png_7_conv4_1200_64_5_1_400_200_512_1_1.0_1.0_2_head_ic.pkl
2020-12-27 13:17:46 Test 1200 .......
2020-12-27 13:17:58 Epoch: 1200 Train 0.1962/0.4348 0.0000
2020-12-27 13:17:58 Epoch: 1200 Val   0.3407/0.6923 0.0000
2020-12-27 13:17:58 Epoch: 1200 Test  0.2469/0.5771 0.0000
2020-12-27 13:18:31 Train 1200 Accuracy: 0.343
2020-12-27 13:19:06 Val   1200 Accuracy: 0.31266666666666665
2020-12-27 13:19:38 Test1 1200 Accuracy: 0.36088888888888887
2020-12-27 13:22:00 Test2 1200 Accuracy: 0.3556
2020-12-27 13:33:04 episode=1200, Test accuracy=0.3560888888888889
2020-12-27 13:33:04 episode=1200, Test accuracy=0.35142222222222225
2020-12-27 13:33:04 episode=1200, Test accuracy=0.35404444444444444
2020-12-27 13:33:04 episode=1200, Test accuracy=0.35531111111111113
2020-12-27 13:33:04 episode=1200, Test accuracy=0.35860000000000003
2020-12-27 13:33:04 episode=1200, Mean Test accuracy=0.35509333333333337


2020-12-28 20:40:42 load matching net success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/3_256_png_7_resnet12_1200_32_5_1_400_200_512_1_1.0_1.0_2_head_mn.pkl
2020-12-28 20:40:42 load ic model success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/3_256_png_7_resnet12_1200_32_5_1_400_200_512_1_1.0_1.0_2_head_ic.pkl
2020-12-28 20:40:42 Test 1200 .......
2020-12-28 20:41:01 Epoch: 1200 Train 0.2031/0.4345 0.0000
2020-12-28 20:41:01 Epoch: 1200 Val   0.3391/0.6877 0.0000
2020-12-28 20:41:01 Epoch: 1200 Test  0.2445/0.5706 0.0000
2020-12-28 20:41:51 Train 1200 Accuracy: 0.35533333333333333
2020-12-28 20:42:44 Val   1200 Accuracy: 0.3177777777777778
2020-12-28 20:43:28 Test1 1200 Accuracy: 0.36866666666666664
2020-12-28 20:46:51 Test2 1200 Accuracy: 0.3676666666666667
2020-12-28 21:01:58 episode=1200, Test accuracy=0.36546666666666666
2020-12-28 21:01:58 episode=1200, Test accuracy=0.3601111111111111
2020-12-28 21:01:58 episode=1200, Test accuracy=0.3602444444444445
2020-12-28 21:01:58 episode=1200, Test accuracy=0.36568888888888895
2020-12-28 21:01:58 episode=1200, Test accuracy=0.36695555555555553
2020-12-28 21:01:58 episode=1200, Mean Test accuracy=0.3636933333333333


2020-12-29 14:03:04 load matching net success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_256_png_7_resnet12_1200_32_5_1_400_200_512_1_1.0_1.0_2_head_mn.pkl
2020-12-29 14:03:04 load ic model success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/0_256_png_7_resnet12_1200_32_5_1_400_200_512_1_1.0_1.0_2_head_ic.pkl
2020-12-29 14:03:04 Test 1200 .......
2020-12-29 14:03:22 Epoch: 1200 Train 0.2065/0.4458 0.0000
2020-12-29 14:03:22 Epoch: 1200 Val   0.3353/0.6978 0.0000
2020-12-29 14:03:22 Epoch: 1200 Test  0.2584/0.5834 0.0000
2020-12-29 14:04:11 Train 1200 Accuracy: 0.36844444444444446
2020-12-29 14:05:09 Val   1200 Accuracy: 0.3268888888888889
2020-12-29 14:06:08 Test1 1200 Accuracy: 0.37577777777777777
2020-12-29 14:09:32 Test2 1200 Accuracy: 0.3732666666666667
2020-12-29 14:27:01 episode=1200, Test accuracy=0.3685555555555555
2020-12-29 14:27:01 episode=1200, Test accuracy=0.36062222222222223
2020-12-29 14:27:01 episode=1200, Test accuracy=0.3661111111111111
2020-12-29 14:27:01 episode=1200, Test accuracy=0.36920000000000003
2020-12-29 14:27:01 episode=1200, Test accuracy=0.3665777777777778
2020-12-29 14:27:01 episode=1200, Mean Test accuracy=0.36621333333333334



2020-12-30 01:40:26 load matching net success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/2_256_png_7_conv4_1200_64_5_1_400_200_2048_1_1.0_1.0_5_head_mn.pkl
2020-12-30 01:40:26 load ic model success from ../cars/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli/2_256_png_7_conv4_1200_64_5_1_400_200_2048_1_1.0_1.0_5_head_ic.pkl
2020-12-30 01:40:26 Test 1200 .......
2020-12-30 01:40:38 Epoch: 1200 Train 0.2602/0.5364 0.0000
2020-12-30 01:40:38 Epoch: 1200 Val   0.3993/0.7413 0.0000
2020-12-30 01:40:38 Epoch: 1200 Test  0.3068/0.6428 0.0000
2020-12-30 01:41:12 Train 1200 Accuracy: 0.35422222222222227
2020-12-30 01:41:45 Val   1200 Accuracy: 0.3115555555555556
2020-12-30 01:42:18 Test1 1200 Accuracy: 0.3596666666666667
2020-12-30 01:44:27 Test2 1200 Accuracy: 0.3572666666666667
2020-12-30 01:55:43 episode=1200, Test accuracy=0.3600444444444444
2020-12-30 01:55:43 episode=1200, Test accuracy=0.35531111111111113
2020-12-30 01:55:43 episode=1200, Test accuracy=0.35928888888888894
2020-12-30 01:55:43 episode=1200, Test accuracy=0.36004444444444444
2020-12-30 01:55:43 episode=1200, Test accuracy=0.36062222222222223
2020-12-30 01:55:43 episode=1200, Mean Test accuracy=0.3590622222222222
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
