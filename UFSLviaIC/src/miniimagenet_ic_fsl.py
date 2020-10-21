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
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


##############################################################################################################


class MiniImageNetIC(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]
        self.transform = transforms.Compose([transforms.CenterCrop(size=84), transforms.ToTensor(),
                                             transforms.Normalize(mean=Config.MEAN_PIXEL, std=Config.STD_PIXEL)])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = image if self.transform is None else self.transform(image)
        return image, label, idx

    pass


class MiniImageNetDataset(object):

    def __init__(self, data_list, is_train, num_way, num_shot):
        self.data_list, self.is_train = data_list, is_train
        self.num_way, self.num_shot = num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        normalize = transforms.Normalize(mean=Config.MEAN_PIXEL, std=Config.STD_PIXEL)
        self.transform_train_ic = transforms.Compose([
            transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_train_fsl = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.CenterCrop(size=84), transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def get_data_all(data_root):
        train_folder = os.path.join(data_root, "train")
        val_folder = os.path.join(data_root, "val")
        test_folder = os.path.join(data_root, "test")

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

        count_image, count_class, data_val_list = 0, 0, []
        for label in os.listdir(val_folder):
            now_class_path = os.path.join(val_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_val_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        count_image, count_class, data_test_list = 0, 0, []
        for label in os.listdir(test_folder):
            now_class_path = os.path.join(test_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_test_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        return data_train_list, data_val_list, data_test_list

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
            transform = self.transform_test
            if self.is_train:
                transform = self.transform_train_ic if one[2] == now_image_filename else self.transform_train_fsl
                pass
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

    pass


##############################################################################################################


class CNNEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        pass

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class RelationNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 3 * 3, 8)
        self.fc2 = nn.Linear(8, 1)
        pass

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out = out2.view(out2.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

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


class ICModel(nn.Module):

    def __init__(self, in_dim, out_dim, linear_bias=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(in_dim, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(in_dim, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.linear = nn.Linear(in_dim, out_dim, bias=linear_bias)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out_logits = self.linear(out)
        out_l2norm = self.l2norm(out_logits)
        out_sigmoid = torch.sigmoid(out_logits)
        return out_logits, out_l2norm, out_sigmoid

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
        self.class_num = np.zeros(shape=(self.out_dim, ), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample, ), dtype=np.int)
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
        top_k = out.data.topk(self.out_dim, dim=1)[1].cpu()
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


class KNN(object):

    @staticmethod
    def knn(feature_encoder, ic_model, low_dim, train_loader, k, t=0.1):

        with torch.no_grad():

            def _cal(_labels, _dist, _train_labels, _retrieval_1_hot, _top1, _top5, _max_c):
                # ---------------------------------------------------------------------------------- #
                _batch_size = _labels.size(0)
                _yd, _yi = _dist.topk(k+1, dim=1, largest=True, sorted=True)
                _yd, _yi = _yd[:, 1:], _yi[:, 1:]
                _candidates = _train_labels.view(1, -1).expand(_batch_size, -1)
                _retrieval = torch.gather(_candidates, 1, _yi)

                _retrieval_1_hot.resize_(_batch_size * k, _max_c).zero_()
                _retrieval_1_hot = _retrieval_1_hot.scatter_(1, _retrieval.view(-1, 1), 1).view(_batch_size, -1, _max_c)
                _yd_transform = _yd.clone().div_(t).exp_().view(_batch_size, -1, 1)
                _probs = torch.sum(torch.mul(_retrieval_1_hot, _yd_transform), 1)
                _, _predictions = _probs.sort(1, True)
                # ---------------------------------------------------------------------------------- #

                _correct = _predictions.eq(_labels.data.view(-1, 1))

                _top1 += _correct.narrow(1, 0, 1).sum().item()
                _top5 += _correct.narrow(1, 0, 5).sum().item()
                return _top1, _top5, _retrieval_1_hot

            n_sample = train_loader.dataset.__len__()
            out_memory = cuda(torch.zeros(n_sample, low_dim).t())
            train_labels = cuda(torch.LongTensor(train_loader.dataset.train_label))
            max_c = train_labels.max() + 1

            out_list = []
            for batch_idx, (inputs, labels, indexes) in enumerate(train_loader):
                sample_features = feature_encoder(cuda(inputs))  # 5x64*19*19
                _, out_l2norm, _ = ic_model(sample_features)
                out_list.append([out_l2norm, cuda(labels)])
                out_memory[:, batch_idx * inputs.size(0):(batch_idx + 1) * inputs.size(0)] = out_l2norm.data.t()
                pass

            top1, top5, total = 0., 0., 0
            retrieval_one_hot = cuda(torch.zeros(k, max_c))  # [200, 10]
            for out in out_list:
                dist = torch.mm(out[0], out_memory)
                total += out[1].size(0)
                top1, top5, retrieval_one_hot = _cal(out[1], dist, train_labels, retrieval_one_hot, top1, top5, max_c)
                pass

            return top1 / total, top5 / total

        pass

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # all data
        self.data_train, self.data_val, self.data_test = MiniImageNetDataset.get_data_all(Config.data_root)

        self.task_train = MiniImageNetDataset(self.data_train, True, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, shuffle=True, num_workers=Config.num_workers)

        self.task_test_train = MiniImageNetDataset(self.data_train, False, Config.num_way, Config.num_shot)
        self.task_test_val = MiniImageNetDataset(self.data_val, False, Config.num_way, Config.num_shot)
        self.task_test_test = MiniImageNetDataset(self.data_test, False, Config.num_way, Config.num_shot)
        self.task_test_train_loader = DataLoader(self.task_test_train, Config.batch_size, shuffle=False, num_workers=Config.num_workers)
        self.task_test_val_loader = DataLoader(self.task_test_val, Config.batch_size, shuffle=False, num_workers=Config.num_workers)
        self.task_test_test_loader = DataLoader(self.task_test_test, Config.batch_size, shuffle=False, num_workers=Config.num_workers)

        self.ic_test_train_loader = DataLoader(MiniImageNetIC(self.data_train), Config.batch_size, shuffle=False, num_workers=Config.num_workers)
        self.ic_test_val_loader = DataLoader(MiniImageNetIC(self.data_val), Config.batch_size, shuffle=False, num_workers=Config.num_workers)
        self.ic_test_test_loader = DataLoader(MiniImageNetIC(self.data_test), Config.batch_size, shuffle=False, num_workers=Config.num_workers)

        # model
        self.feature_encoder = cuda(CNNEncoder())
        self.relation_network = cuda(RelationNetwork())
        self.ic_model = cuda(ICModel(in_dim=Config.ic_in_dim, out_dim=Config.ic_out_dim))

        # loss
        self.ic_loss = cuda(nn.CrossEntropyLoss())
        self.fsl_loss = cuda(nn.MSELoss())

        # optim
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=Config.learning_rate)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=Config.learning_rate)
        self.ic_model_optim = torch.optim.Adam(self.ic_model.parameters(), lr=Config.learning_rate)

        # DHC
        self.produce_class1 = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class2 = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class1.init()
        self.produce_class2.init()
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

    def train(self):
        Tools.print()
        Tools.print("Training...")

        for epoch in range(Config.train_epoch):
            self.feature_encoder.train()
            self.relation_network.train()
            self.ic_model.train()

            Tools.print()
            self.produce_class1.reset()
            all_loss, all_loss_fsl, all_loss_ic = 0.0, 0.0, 0.0
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels, ic_labels = cuda(task_data), cuda(task_labels), cuda(task_index[:, -1])

                ###########################################################################
                # 1 calculate features
                relations, query_features = self.compare_fsl(task_data)
                ic_out_logits, ic_out_l2norm, ic_out_sigmoid = self.ic_model(query_features)

                # 2
                self.produce_class1.cal_label(ic_out_l2norm, ic_labels)
                ic_targets = self.produce_class2.get_label(ic_labels)

                # 3 loss
                loss_fsl = self.fsl_loss(relations, task_labels) * 1.0
                loss_ic = self.ic_loss(ic_out_logits, ic_targets) * 0.1
                loss = loss_fsl + loss_ic
                all_loss += loss.item()
                all_loss_fsl += loss_fsl.item()
                all_loss_ic += loss_ic.item()

                # 4 backward
                self.feature_encoder.zero_grad()
                self.relation_network.zero_grad()
                self.ic_model.zero_grad()
                loss.backward()
                self.feature_encoder_optim.step()
                self.relation_network_optim.step()
                self.ic_model_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} fsl:{:.3f} ic:{:.3f} lr:{}".format(
                epoch + 1, all_loss / len(self.task_train_loader), all_loss_fsl / len(self.task_train_loader),
                all_loss_ic / len(self.task_train_loader), Config.learning_rate))
            ###########################################################################

            ###########################################################################
            # 切换
            classes = self.produce_class2.classes
            self.produce_class2.classes = self.produce_class1.classes
            self.produce_class1.classes = classes
            Tools.print("Train: [{}] {}/{}".format(
                epoch, self.produce_class1.count, self.produce_class1.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                self.feature_encoder.eval()
                self.relation_network.eval()
                self.ic_model.eval()

                # val ic
                Tools.print()
                Tools.print("Test {} .......".format(epoch))
                self.val_ic(epoch, ic_loader=self.ic_test_train_loader, name="Train")
                self.val_ic(epoch, ic_loader=self.ic_test_val_loader, name="Val")
                self.val_ic(epoch, ic_loader=self.ic_test_test_loader, name="Test")

                # val fsl
                self.val_fsl(epoch, self.task_test_train_loader, name="Train")
                val_accuracy = self.val_fsl(epoch, self.task_test_val_loader, name="Val")
                self.val_fsl(epoch, loader=self.task_test_test_loader, name="Test")
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

    def val_ic(self, epoch, ic_loader, name="Test"):
        acc_1, acc_2 = KNN.knn(self.feature_encoder, self.ic_model, Config.ic_out_dim, ic_loader, 100)
        Tools.print("Epoch: [{}] {} {:.4f}/{:.4f}".format(epoch, name, acc_1, acc_2))
        pass

    def val_fsl(self, epoch, loader, name=None):
        accuracies = []
        total_rewards, counter = 0, 0
        for task_data, task_labels, _ in tqdm(loader):
            task_data, task_labels = cuda(task_data), cuda(task_labels)
            batch_size = task_labels.shape[0]

            relations, _ = self.compare_fsl(task_data)
            _, predict_labels = torch.max(relations.data, 1)
            rewards = [int(task_labels[i][predict_labels[i]]) for i in range(batch_size)]

            total_rewards += np.sum(rewards)
            counter += batch_size
            accuracies.append(total_rewards / 1.0 / counter)
            pass

        final_accuracy = np.mean(np.array(accuracies, dtype=np.float))
        Tools.print("Val {} {} Accuracy: {}".format(epoch, name, final_accuracy))
        return final_accuracy

    def val_fsl_test(self, epoch, test_avg_num=2):
        Tools.print()
        Tools.print("Testing...")
        total_accuracy = 0.0
        for _which in range(test_avg_num):
            test_accuracy = self.val_fsl(epoch, loader=self.task_test_test_loader, name="Test {}".format(_which))
            total_accuracy += test_accuracy
            pass
        final_accuracy = total_accuracy / Config.test_avg_num
        Tools.print("Final Test accuracy: {}".format(final_accuracy))
        return final_accuracy

    def compare_fsl(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        fe_inputs = task_data.view([-1, data_num_channel, data_width, data_weight])  # 90, 3, 84, 84

        # feature encoder
        data_features = self.feature_encoder(fe_inputs)  # 90x64*19*19
        _, feature_dim, feature_width, feature_height = data_features.shape
        data_features = data_features.view([-1, data_image_num, feature_dim, feature_width, feature_height])
        data_features_support, data_features_query = data_features.split(Config.num_shot * Config.num_way, dim=1)
        data_features_query_repeat = data_features_query.repeat(1, Config.num_shot * Config.num_way, 1, 1, 1)

        # calculate relations
        relation_pairs = torch.cat((data_features_support, data_features_query_repeat), 2)
        relation_pairs = relation_pairs.view(-1, feature_dim * 2, feature_width, feature_height)
        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, Config.num_way * Config.num_shot)

        return relations, data_features_query.squeeze()

    pass


##############################################################################################################


"""
020-10-21 23:02:50 Train: [999] 19471/1854
2020-10-21 23:02:50 load feature encoder success from ../models/ic_fsl/2_64_5_1_fe_5way_1shot.pkl
2020-10-21 23:02:50 load relation network success from ../models/ic_fsl/2_64_5_1_rn_5way_1shot.pkl
2020-10-21 23:02:50 load ic model success from ../models/ic_fsl/2_64_5_1_ic_5way_1shot.pkl
2020-10-21 23:02:58 Epoch: [1000] Final Train 0.3397/0.6713
2020-10-21 23:03:00 Epoch: [1000] Final Val 0.4889/0.8769
2020-10-21 23:03:02 Epoch: [1000] Final Test 0.4594/0.8553
2020-10-21 23:04:09 Val 1000 Final Train Accuracy: 0.5360503222390252
2020-10-21 23:04:16 Val 1000 Final Val   Accuracy: 0.5810896711451617
2020-10-21 23:04:26 Val 1000 Final Test  Accuracy: 0.4569410663324587
"""


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train_epoch = 1000
    learning_rate = 0.001
    num_workers = 8

    val_freq = 10

    num_way = 5
    num_shot = 1
    batch_size = 64
    test_avg_num = 2

    model_name = "2_{}_{}_{}".format(batch_size, num_way, num_shot)

    # ic
    ic_in_dim = 64
    ic_out_dim = 512
    ic_ratio = 2

    MEAN_PIXEL = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    STD_PIXEL = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    fe_dir = Tools.new_dir("../models/ic_fsl/{}_fe_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    rn_dir = Tools.new_dir("../models/ic_fsl/{}_rn_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    ic_dir = Tools.new_dir("../models/ic_fsl/{}_ic_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    pass


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.feature_encoder.eval()
    runner.relation_network.eval()
    runner.ic_model.eval()
    runner.val_ic(0, ic_loader=runner.ic_test_train_loader, name="Train")
    runner.val_ic(0, ic_loader=runner.ic_test_val_loader, name="Val")
    runner.val_ic(0, ic_loader=runner.ic_test_test_loader, name="Test")
    runner.val_fsl(epoch=0, loader=runner.task_test_train_loader, name="First Train")
    runner.val_fsl(epoch=0, loader=runner.task_test_val_loader, name="First Val")
    runner.val_fsl(epoch=0, loader=runner.task_test_test_loader, name="First Test")

    runner.train()

    runner.load_model()
    runner.feature_encoder.eval()
    runner.relation_network.eval()
    runner.ic_model.eval()
    runner.val_ic(epoch=Config.train_epoch, ic_loader=runner.ic_test_train_loader, name="Final Train")
    runner.val_ic(epoch=Config.train_epoch, ic_loader=runner.ic_test_val_loader, name="Final Val")
    runner.val_ic(epoch=Config.train_epoch, ic_loader=runner.ic_test_test_loader, name="Final Test")
    runner.val_fsl(epoch=Config.train_epoch, loader=runner.task_test_train_loader, name="Final Train")
    runner.val_fsl(epoch=Config.train_epoch, loader=runner.task_test_val_loader, name="Final Val")
    runner.val_fsl_test(epoch=Config.train_epoch, test_avg_num=Config.test_avg_num)
    pass
