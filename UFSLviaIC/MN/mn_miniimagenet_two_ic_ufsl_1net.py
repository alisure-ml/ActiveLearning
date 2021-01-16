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
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34
from mn_miniimagenet_fsl_test_tool import TestTool
from mn_miniimagenet_ic_test_tool import ICTestTool
from mn_miniimagenet_tool import MatchingNet, ResNet12Small, BasicBlock, Normalize, RunnerTool


##############################################################################################################


class MiniImageNetDataset(object):

    def __init__(self, data_list, num_way, num_shot):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot
        self.data_id = np.asarray(range(len(self.data_list)))

        self.classes = None

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
            transform = self.transform_train_ic if one[2] == now_image_filename else self.transform_train_fsl
            task_data.append(torch.unsqueeze(self.read_image(one[2], transform), dim=0))
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


class EncoderC4(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = MatchingNet(64, 64)
        self.out_dim = 1600
        pass

    def forward(self, x):
        out = self.encoder(x)
        out = torch.flatten(out, 1)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class EncoderResNet12(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet12Small(avg_pool=True, drop_rate=0.1)
        self.out_dim = 512
        pass

    def forward(self, x):
        out = self.encoder(x)
        out = torch.flatten(out, 1)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class EncoderResNet34(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = resnet34(num_classes=512)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_dim = 512
        pass

    def forward(self, x):
        out = self.forward_impl(x)
        out = torch.flatten(out, 1)
        return out

    def forward_impl(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        return x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class OneNet(nn.Module):

    def __init__(self, encoder, low_dim=512):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(self.encoder.out_dim, low_dim)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x, only_fsl=False, only_ic=False, all_out=False):
        out_fsl = self.encoder(x)

        if only_fsl:
            return out_fsl

        out_ic = self.fc(out_fsl)
        out_ic_l2norm = self.l2norm(out_ic)

        if only_ic:
            return out_ic, out_ic_l2norm

        if all_out:
            return out_fsl, out_ic, out_ic_l2norm

        return out_ic, out_ic_l2norm

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


class Runner(object):

    def __init__(self):
        self.adjust_learning_rate = Config.adjust_learning_rate

        # all data
        self.data_train = MiniImageNetDataset.get_data_all(Config.data_root)
        self.task_train = MiniImageNetDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()
        self.task_train.set_samples_class(self.produce_class.classes)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        self.norm = Normalize(2)
        if Config.multi_gpu:
            self.matching_net = RunnerTool.to_cuda(nn.DataParallel(self.matching_net))
            cudnn.benchmark = True
            pass
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))

        # optim
        self.matching_net_optim = torch.optim.SGD(
            self.matching_net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # loss
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())
        self.fsl_loss = RunnerTool.to_cuda(nn.MSELoss())

        # Eval
        self.test_tool_fsl = TestTool(self.matching_test, data_root=Config.data_root,
                                      num_way=Config.num_way, num_shot=Config.num_shot,
                                      episode_size=Config.episode_size, test_episode=Config.test_episode,
                                      transform=self.task_train.transform_test)
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.matching_net,
                                       data_root=Config.data_root, batch_size=Config.batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load matching net success from {}".format(Config.mn_dir))
        pass

    def matching(self, task_data):  # task_data[:, -1]
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out, ic_out_logits, ic_out_l2norm = self.matching_net(data_x, all_out=True)
        z = net_out.view(data_batch_size, data_image_num, -1)
        ic_out_logits = ic_out_logits.view(data_batch_size, data_image_num, -1)[:, -1]
        ic_out_l2norm = ic_out_l2norm.view(data_batch_size, data_image_num, -1)[:, -1]

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
        return predicts, ic_out_logits, ic_out_l2norm

    def matching_test(self, samples, batches):
        if Config.multi_gpu:
            device_ids = self.matching_net.device_ids
            self.matching_net.device_ids = self.matching_net.device_ids[0]
            pass

        batch_num, _, _, _ = batches.shape
        sample_z = self.matching_net(samples, only_fsl=True)  # 5x64*5*5
        batch_z = self.matching_net(batches, only_fsl=True)  # 75x64*5*5
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

        if Config.multi_gpu:
            self.matching_net.device_ids = device_ids
            pass
        return predicts

    def train(self):
        Tools.print()
        Tools.print("Training...")

        # Init Update
        try:
            self.matching_net.eval()
            Tools.print("Init label {} .......")
            self.produce_class.reset()
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)
                ic_out_logits, ic_out_l2norm = self.matching_net(task_data[:, -1], only_ic=True)
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2))
        finally:
            pass

        best_accuracy = 0.0
        for epoch in range(1, 1 + Config.train_epoch):
            self.matching_net.train()

            Tools.print()
            mn_lr= self.adjust_learning_rate(self.matching_net_optim, epoch,
                                             Config.first_epoch, Config.t_epoch, Config.learning_rate)
            Tools.print('Epoch: [{}] mn_lr={}'.format(epoch, mn_lr))

            self.produce_class.reset()
            Tools.print(self.task_train.classes)
            is_ok_total, is_ok_acc = 0, 0
            all_loss, all_loss_fsl, all_loss_ic = 0.0, 0.0, 0.0
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                ic_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                relations, ic_out_logits, ic_out_l2norm = self.matching(task_data)

                # 2
                ic_targets = self.produce_class.get_label(ic_labels)
                self.produce_class.cal_label(ic_out_l2norm, ic_labels)

                # 3 loss
                loss_fsl = self.fsl_loss(relations, task_labels)
                loss_ic = self.ic_loss(ic_out_logits, ic_targets)
                loss = loss_fsl + loss_ic
                all_loss += loss.item()
                all_loss_fsl += loss_fsl.item()
                all_loss_ic += loss_ic.item()

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
            Tools.print("{:6} loss:{:.3f} fsl:{:.3f} ic:{:.3f} ok:{:.3f}({}/{})".format(
                epoch + 1, all_loss / len(self.task_train_loader),
                all_loss_fsl / len(self.task_train_loader), all_loss_ic / len(self.task_train_loader),
                int(is_ok_acc) / int(is_ok_total), is_ok_acc, is_ok_total, ))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                self.matching_net.eval()

                self.test_tool_ic.val(epoch=epoch)
                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True)

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
    multi_gpu = True

    gpu_id = "0,1,2,3" if multi_gpu else "1"
    gpu_num = len(gpu_id.split(","))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    num_workers = 8

    num_way = 5
    num_shot = 1

    val_freq = 10
    episode_size = 15
    test_episode = 600

    # ic
    ic_out_dim = 512
    ic_ratio = 1

    learning_rate = 0.01

    train_epoch = 1500
    first_epoch, t_epoch = 500, 200
    adjust_learning_rate = RunnerTool.adjust_learning_rate1

    ###############################################################################################
    # matching_net, batch_size, net_name = OneNet(encoder=EncoderC4(), low_dim=ic_out_dim), 64, "C4"
    matching_net, batch_size, net_name = OneNet(encoder=EncoderResNet12(), low_dim=ic_out_dim), 32, "R12S"
    # matching_net, batch_size, net_name = OneNet(encoder=EncoderResNet34(), low_dim=ic_out_dim), 32, "R34"
    ###############################################################################################
    batch_size = batch_size * gpu_num

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_png".format(
        gpu_id.replace(",", ""), net_name, train_epoch, batch_size,
        num_way, num_shot, first_epoch, t_epoch, ic_out_dim, ic_ratio)
    Tools.print(model_name)

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"
    data_root = os.path.join(data_root, "miniImageNet_png")
    Tools.print(data_root)

    _root_path = "../models_abl/mn/two_ic_ufsl_1net"
    mn_dir = Tools.new_dir("{}/{}.pkl".format(_root_path, model_name))
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.matching_net.eval()
    runner.test_tool_ic.val(epoch=0, is_print=True)
    runner.test_tool_fsl.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.matching_net.eval()
    runner.test_tool_ic.val(epoch=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
