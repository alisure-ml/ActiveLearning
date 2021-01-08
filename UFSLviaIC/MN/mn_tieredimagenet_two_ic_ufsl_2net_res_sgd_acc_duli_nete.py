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
import torch.backends.cudnn as cudnn
from mn_tool_ic_test import ICTestTool
from mn_tool_fsl_test import FSLTestTool
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from mn_tool_net import MatchingNet, Normalize, RunnerTool, ResNet12Small


##############################################################################################################


class TieredImageNetICDataset(object):

    def __init__(self, data_list, image_size=84):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                         std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.transform_train_ic = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = self.transform_train_ic(image)
        return image, label, idx

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


class TieredImageNetFSLDatasetOld(object):

    def __init__(self, data_list, classes, num_way, num_shot, image_size=84):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot
        self.data_id = np.asarray(range(len(self.data_list)))
        self.classes = classes

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                         std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.transform_train_fsl = transforms.Compose([
            transforms.RandomCrop(image_size, padding=8),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, _, now_image_filename = now_label_image_tuple
        _now_label = self.classes[item]

        now_label_k_shot_index = self._get_samples_by_clustering_label(_now_label, True, num=self.num_shot)
        is_ok_list = [self.data_list[one][1] == now_label_image_tuple[1] for one in now_label_k_shot_index]

        # 其他样本
        other_label_k_shot_index_list = self._get_samples_by_clustering_label(
            _now_label, False, num=self.num_shot * (self.num_way - 1))

        # c_way_k_shot
        c_way_k_shot_index_list = now_label_k_shot_index + other_label_k_shot_index_list
        random.shuffle(c_way_k_shot_index_list)

        if len(c_way_k_shot_index_list) != self.num_shot * self.num_way:
            return self.__getitem__(random.sample(list(range(0, len(self.data_list))), 1)[0])

        task_list = [self.data_list[index] for index in c_way_k_shot_index_list] + [now_label_image_tuple]

        task_data = []
        for one in task_list:
            task_data.append(torch.unsqueeze(self.read_image(one, self.transform_train_fsl), dim=0))
            pass
        task_data = torch.cat(task_data)

        task_label = torch.Tensor([int(index in now_label_k_shot_index) for index in c_way_k_shot_index_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index, is_ok_list

    def _get_samples_by_clustering_label(self, label, is_same_label=False, num=1):
        if is_same_label:
            return random.sample(list(np.squeeze(np.argwhere(self.classes == label), axis=1)), num)
        else:
            return random.sample(list(np.squeeze(np.argwhere(self.classes != label))), num)
        pass

    @staticmethod
    def read_image(one, transform=None):
        image = Image.open(one[2]).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    pass


class TieredImageNetFSLDataset(object):

    def __init__(self, data_list, classes, num_way, num_shot, image_size=84):
        self.num_way, self.num_shot = num_way, num_shot
        self.true_label = [one[1] for one in data_list]
        self.data_list = [(one[0], class_one, one[2]) for one, class_one in zip(data_list, classes)]

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass
        Tools.print("class number = {}".format(len(self.data_dict.keys())))

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                         std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.transform_train_fsl = transforms.Compose([
            transforms.RandomCrop(image_size, padding=8),
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
        is_ok_list = [self.true_label[one[0]] == self.true_label[item] for one in now_label_k_shot_image_tuple]

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
        task_data = torch.cat([torch.unsqueeze(self.read_image(one, self.transform_train_fsl),
                                               dim=0) for one in task_list])
        task_label = torch.Tensor([int(one_tuple[1] == now_label) for one_tuple in c_way_k_shot_tuple_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index, is_ok_list

    @staticmethod
    def read_image(one, transform=None):
        image = Image.open(one[2]).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

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


class RunnerIC(object):

    def __init__(self):
        self.adjust_learning_rate = Config.ic_adjust_learning_rate

        # data
        self.data_train = TieredImageNetICDataset.get_data_all(Config.data_root)
        self.tiered_imagenet_dataset = TieredImageNetICDataset(self.data_train)
        self.ic_train_loader = DataLoader(self.tiered_imagenet_dataset, Config.ic_batch_size,
                                          shuffle=True, num_workers=Config.num_workers)
        self.ic_train_loader_eval = DataLoader(self.tiered_imagenet_dataset, Config.ic_batch_size,
                                               shuffle=False, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()

        # model
        self.ic_model = RunnerTool.to_cuda(ICResNet(Config.ic_resnet, low_dim=Config.ic_out_dim,
                                                    modify_head=Config.ic_modify_head))
        self.ic_model = RunnerTool.to_cuda(nn.DataParallel(self.ic_model))
        cudnn.benchmark = True
        RunnerTool.to_cuda(self.ic_model.apply(RunnerTool.weights_init))
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())

        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=Config.ic_learning_rate, momentum=0.9, weight_decay=5e-4)

        # Eval
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
                                       data_root=Config.data_root, batch_size=Config.ic_batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim,
                                       transform=self.ic_train_loader.dataset.transform_test, k=Config.ic_knn)
        pass

    def load_model(self, ic_dir_checkpoint=None):
        ic_dir = ic_dir_checkpoint if ic_dir_checkpoint else Config.ic_dir
        if os.path.exists(ic_dir):
            checkpoint = torch.load(ic_dir)
            if ic_dir_checkpoint:
                if not list(self.ic_model.state_dict().keys())[0] == list(checkpoint.keys())[0]:
                    checkpoint = {"module.{}".format(key): checkpoint[key] for key in checkpoint}
            self.ic_model.load_state_dict(checkpoint)
            Tools.print("load ic model success from {}".format(ic_dir))
        pass

    def train(self):
        Tools.print()
        Tools.print("Training...")
        best_accuracy = 0.0

        # Init Update
        try:
            self.ic_model.eval()
            Tools.print("Init label {} .......")
            self.produce_class.reset()
            with torch.no_grad():
                for image, label, idx in tqdm(self.ic_train_loader):
                    image, idx = RunnerTool.to_cuda(image), RunnerTool.to_cuda(idx)
                    ic_out_logits, ic_out_l2norm = self.ic_model(image)
                    self.produce_class.cal_label(ic_out_l2norm, idx)
                    pass
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2))
        finally:
            pass

        for epoch in range(1, 1 + Config.ic_train_epoch):
            self.ic_model.train()

            Tools.print()
            ic_lr = self.adjust_learning_rate(self.ic_model_optim, epoch, Config.ic_first_epoch,
                                              Config.ic_t_epoch, Config.ic_learning_rate)
            Tools.print('Epoch: [{}] ic_lr={}'.format(epoch, ic_lr))

            all_loss = 0.0
            self.produce_class.reset()
            for image, label, idx in tqdm(self.ic_train_loader):
                image, label, idx = RunnerTool.to_cuda(image), RunnerTool.to_cuda(label), RunnerTool.to_cuda(idx)

                ###########################################################################
                # 1 calculate features
                ic_out_logits, ic_out_l2norm = self.ic_model(image)

                # 2 calculate labels
                ic_targets = self.produce_class.get_label(idx)
                self.produce_class.cal_label(ic_out_l2norm, idx)

                # 3 loss
                loss = self.ic_loss(ic_out_logits, ic_targets)
                all_loss += loss.item()

                # 4 backward
                self.ic_model.zero_grad()
                loss.backward()
                self.ic_model_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f}".format(epoch, all_loss / len(self.ic_train_loader)))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.ic_val_freq == 0:
                self.ic_model.eval()

                val_accuracy = self.test_tool_ic.val(epoch=epoch)
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.ic_model.state_dict(), Config.ic_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    def eval(self):
        self.ic_model.eval()
        with torch.no_grad():
            ic_classes = np.zeros(shape=(len(self.tiered_imagenet_dataset),), dtype=np.int)
            for image, label, idx in tqdm(self.ic_train_loader_eval):
                ic_out_logits, ic_out_l2norm = self.ic_model(RunnerTool.to_cuda(image))
                ic_classes[idx] = np.argmax(ic_out_l2norm.cpu().numpy(), axis=-1)
                pass
        return self.data_train, ic_classes

    pass


class RunnerFSL(object):

    def __init__(self, data_train, classes):
        # all data
        self.data_train = data_train
        self.task_train = TieredImageNetFSLDataset(self.data_train, classes, Config.fsl_num_way, Config.fsl_num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.fsl_batch_size, True, num_workers=Config.num_workers)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.fsl_matching_net)
        self.matching_net = RunnerTool.to_cuda(nn.DataParallel(self.matching_net))
        cudnn.benchmark = True
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))
        self.norm = Normalize(2)

        # optim
        self.matching_net_optim = torch.optim.Adam(self.matching_net.parameters(), lr=Config.fsl_learning_rate)
        self.matching_net_scheduler = MultiStepLR(self.matching_net_optim, Config.fsl_lr_schedule, gamma=1/3)

        # loss
        self.fsl_loss = RunnerTool.to_cuda(nn.MSELoss())

        # Eval
        self.test_tool_fsl = FSLTestTool(self.matching_test, data_root=Config.data_root,
                                         num_way=Config.fsl_num_way, num_shot=Config.fsl_num_shot,
                                         episode_size=Config.fsl_episode_size, test_episode=Config.fsl_test_episode,
                                         transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load matching net success from {}".format(Config.mn_dir))
        pass

    def matching(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.matching_net(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        # 特征
        z_support, z_query = z.split(Config.fsl_num_shot * Config.fsl_num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, Config.fsl_num_way * Config.fsl_num_shot, z_dim)
        z_query_expand = z_query.expand(z_batch_size, Config.fsl_num_way * Config.fsl_num_shot, z_dim)

        # 相似性
        z_support = self.norm(z_support)
        similarities = torch.sum(z_support * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(z_batch_size, Config.fsl_num_way, Config.fsl_num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def matching_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)  # 5x64*5*5
        batch_z = self.matching_net(batches)  # 75x64*5*5
        z_support = sample_z.view(Config.fsl_num_way * Config.fsl_num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, Config.fsl_num_way * Config.fsl_num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, Config.fsl_num_way * Config.fsl_num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, Config.fsl_num_way, Config.fsl_num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def train(self):
        Tools.print()
        Tools.print("Training...")
        best_accuracy = 0.0

        for epoch in range(1, 1 + Config.fsl_train_epoch):
            self.matching_net.train()

            Tools.print()
            all_loss, is_ok_total, is_ok_acc = 0.0, 0, 0
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                relations = self.matching(task_data)

                # 3 loss
                loss = self.fsl_loss(relations, task_labels)
                all_loss += loss.item()

                # 4 backward
                self.matching_net.zero_grad()
                loss.backward()
                self.matching_net_optim.step()

                # is ok
                is_ok_acc += torch.sum(torch.cat(task_ok))
                is_ok_total += torch.prod(torch.tensor(torch.cat(task_ok).shape))
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} ok:{:.3f}({}/{}) lr:{}".format(
                epoch, all_loss / len(self.task_train_loader), int(is_ok_acc) / int(is_ok_total),
                is_ok_acc, is_ok_total, self.matching_net_scheduler.get_last_lr()))
            self.matching_net_scheduler.step()
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.fsl_val_freq == 0:
                self.matching_net.eval()

                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True, has_test=False)

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
    gpu_id = "0,1,2,3"
    gpu_num = len(gpu_id.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 32

    #######################################################################################
    ic_ratio = 1
    ic_knn = 100
    ic_out_dim = 2048
    ic_val_freq = 10

    # ic_resnet, ic_modify_head, ic_net_name = resnet18, False, "res18"
    ic_resnet, ic_modify_head, ic_net_name = resnet34, True, "res34_head"

    ic_learning_rate = 0.01
    ic_train_epoch = 1200
    ic_first_epoch, ic_t_epoch = 400, 200
    ic_batch_size = 64 * 4 * gpu_num

    ic_adjust_learning_rate = RunnerTool.adjust_learning_rate1
    #######################################################################################

    ###############################################################################################
    fsl_num_way = 5
    fsl_num_shot = 1

    fsl_episode_size = 15
    fsl_test_episode = 600

    # fsl_matching_net, fsl_net_name, fsl_batch_size = MatchingNet(hid_dim=64, z_dim=64), "conv4", 96
    fsl_matching_net, fsl_net_name, fsl_batch_size = ResNet12Small(avg_pool=True, drop_rate=0.1), "resnet12", 32

    fsl_learning_rate = 0.01
    fsl_batch_size = fsl_batch_size * gpu_num

    # fsl_val_freq = 5
    # fsl_train_epoch = 200
    # fsl_lr_schedule = [100, 150]

    # fsl_val_freq = 2
    # fsl_train_epoch = 100
    # fsl_lr_schedule = [50, 80]
    fsl_val_freq = 2
    fsl_train_epoch = 50
    fsl_lr_schedule = [30, 40]
    ###############################################################################################

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        gpu_id.replace(",", ""), ic_net_name, ic_train_epoch, ic_batch_size, ic_out_dim,
        fsl_net_name, fsl_train_epoch, fsl_num_way, fsl_num_shot, fsl_batch_size)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/tiered-imagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/tiered-imagenet'
    else:
        data_root = "F:\\data\\tiered-imagenet"

    ###############################################################################################
    # ic_batch_size = 16
    # fsl_batch_size = 16
    # ic_train_epoch = 2
    # ic_first_epoch, ic_t_epoch = 1, 1
    # ic_val_freq = 1
    # fsl_train_epoch = 8
    # fsl_lr_schedule = [4, 6]
    # fsl_test_episode = 20
    # fsl_val_freq = 2
    # data_root = os.path.join(data_root, "small")
    ###############################################################################################

    _root_path = "../tiered_imagenet/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete"
    mn_dir = Tools.new_dir("{}/{}_mn.pkl".format(_root_path, model_name))
    ic_dir = Tools.new_dir("{}/{}_ic.pkl".format(_root_path, model_name))
    # ic_dir_checkpoint = None
    # ic_dir_checkpoint = "../tiered_imagenet/models/ic_res_xx/3_resnet_18_64_2048_1_1900_300_200_False_ic.pkl"
    ic_dir_checkpoint = "../tiered_imagenet/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/123_res34_head_1200_384_2048_conv4_100_5_1_288_ic.pkl"

    Tools.print(model_name)
    Tools.print(data_root)
    pass


"""
2021-01-02 16:22:56 load ic model success from ../tiered_imagenet/models/ic_res_xx/3_resnet_18_64_2048_1_1900_300_200_False_ic.pkl
2021-01-02 16:24:21 class number = 2039
2021-01-02 16:24:21 load matching net success from ../tiered_imagenet/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/0123_res18_1200_1024_2048_conv4_200_5_1_384_mn.pkl
2021-01-02 16:24:56 Train 200 Accuracy: 0.48877777777777776
2021-01-02 16:25:31 Val   200 Accuracy: 0.47944444444444445
2021-01-02 16:26:06 Test1 200 Accuracy: 0.4798888888888888
2021-01-02 16:28:05 Test2 200 Accuracy: 0.46691111111111117
2021-01-02 16:38:08 episode=200, Test accuracy=0.4728444444444445
2021-01-02 16:38:08 episode=200, Test accuracy=0.4697111111111112
2021-01-02 16:38:08 episode=200, Test accuracy=0.47111111111111115
2021-01-02 16:38:08 episode=200, Test accuracy=0.477
2021-01-02 16:38:08 episode=200, Test accuracy=0.4700666666666668
2021-01-02 16:38:08 episode=200, Mean Test accuracy=0.4721466666666667

2021-01-04 03:55:37 load matching net success from ../tiered_imagenet/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/0123_res18_1200_1024_2048_resnet12_100_5_1_128_mn.pkl
2021-01-04 03:56:38 Train 100 Accuracy: 0.5441111111111112
2021-01-04 03:57:38 Val   100 Accuracy: 0.5011111111111112
2021-01-04 03:58:39 Test1 100 Accuracy: 0.5006666666666666
2021-01-04 04:02:17 Test2 100 Accuracy: 0.4976000000000001
2021-01-04 04:20:27 episode=100, Test accuracy=0.4953111111111112
2021-01-04 04:20:27 episode=100, Test accuracy=0.4992888888888889
2021-01-04 04:20:27 episode=100, Test accuracy=0.49140000000000006
2021-01-04 04:20:27 episode=100, Test accuracy=0.4898
2021-01-04 04:20:27 episode=100, Test accuracy=0.48611111111111105
2021-01-04 04:20:27 episode=100, Mean Test accuracy=0.4923822222222222
"""


"""
123_res34_head_1200_384_2048_conv4_100_5_1_288_ic
../tiered_imagenet/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/123_res34_head_1200_384_2048_conv4_100_5_1_288_ic.pkl
2021-01-08 14:50:07 Test 1170 .......
2021-01-08 14:55:13 Epoch: 1170 Train 0.3558/0.6511 0.0000
2021-01-08 14:55:13 Epoch: 1170 Val   0.4054/0.7416 0.0000
2021-01-08 14:55:13 Epoch: 1170 Test  0.3471/0.6674 0.0000
2021-01-08 14:55:13 Save networks for epoch: 1170

123_res34_head_1200_384_2048_conv4_100_5_1_288_mn
../tiered_imagenet/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/123_res34_head_1200_384_2048_conv4_100_5_1_288_mn.pkl
2021-01-09 00:39:45     84 loss:0.037 ok:0.137(61645/448695) lr:[0.001111111111111111]
2021-01-09 00:40:16 Train 84 Accuracy: 0.5206666666666666
2021-01-09 00:40:48 Val   84 Accuracy: 0.501
2021-01-09 00:41:20 Test1 84 Accuracy: 0.49299999999999994
2021-01-09 00:41:20 Save networks for epoch: 84

"""


if __name__ == '__main__':
    runner_ic = RunnerIC()
    # runner_ic.train()
    runner_ic.load_model(ic_dir_checkpoint=Config.ic_dir_checkpoint)
    data_train, classes = runner_ic.eval()

    runner_fsl = RunnerFSL(data_train=data_train, classes=classes)
    runner_fsl.train()

    runner_ic.load_model()
    runner_fsl.load_model()
    runner_ic.ic_model.eval()
    runner_fsl.matching_net.eval()
    # runner_ic.test_tool_ic.val(epoch=Config.ic_train_epoch, is_print=True)
    runner_fsl.test_tool_fsl.val(episode=Config.fsl_train_epoch, is_print=True)
    runner_fsl.test_tool_fsl.test(test_avg_num=5, episode=Config.fsl_train_epoch, is_print=True)
    pass
