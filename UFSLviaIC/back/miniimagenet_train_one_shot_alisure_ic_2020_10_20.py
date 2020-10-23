import os
import math
import torch
import random
import platform
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


##############################################################################################################


class ClassBalancedSampler(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        super().__init__(num_cl)
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
            pass

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
            pass

        return iter(batch)

    def __len__(self):
        return 1

    pass


class ClassBalancedSamplerTest(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_cl, num_inst, shuffle=True):
        super().__init__(num_cl)
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i + j * self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
            pass

        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
                pass
            pass

        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1

    pass


class MiniImageNetTask(object):

    def __init__(self, character_folders, batch_size, num_way, num_shot, train_image_filename_list=None):
        self.character_folders = character_folders
        self.batch_size = batch_size
        self.num_way = num_way
        self.num_shot = num_shot
        self.train_image_filename_list = train_image_filename_list

        # 当前任务所用的数据类别
        class_folders = random.sample(self.character_folders, self.num_way)
        labels = dict(zip(class_folders, np.array(range(len(class_folders)))))

        samples = dict()
        self.train_roots, self.test_roots = [], []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))  # c类所有的数据
            random.shuffle(samples[c])

            self.train_roots += samples[c][:self.num_shot]  # 前K个为训练数据
            self.test_roots += samples[c][self.num_shot:self.num_shot + self.batch_size]  # 后batch_size个测试数据
            pass

        self.train_labels = [labels[os.path.split(x)[0]] for x in self.train_roots]
        self.test_labels = [labels[os.path.split(x)[0]] for x in self.test_roots]
        self.ic_train_ids, self.ic_test_ids = None, None
        if self.train_image_filename_list:
            self.ic_train_ids = [self.train_image_filename_list.index(x) for x in self.train_roots]
            self.ic_test_ids = [self.train_image_filename_list.index(x) for x in self.test_roots]
            pass
        pass

    @staticmethod
    def folders(data_root):
        train_folder = os.path.join(data_root, "train")
        val_folder = os.path.join(data_root, "val")
        test_folder = os.path.join(data_root, "test")

        folders_train = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]
        folders_val = [os.path.join(val_folder, label) for label in os.listdir(val_folder)
                       if os.path.isdir(os.path.join(val_folder, label))]
        folders_test = [os.path.join(test_folder, label) for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))]

        random.seed(1)
        random.shuffle(folders_train)
        random.shuffle(folders_val)
        random.shuffle(folders_test)

        train_image_filename, val_image_filename, test_image_filename = [], [], []
        for folder in folders_train:
            train_image_filename.extend([os.path.join(folder, name) for name in os.listdir(folder)])
        for folder in folders_val:
            val_image_filename.extend([os.path.join(folder, name) for name in os.listdir(folder)])
        for folder in folders_test:
            test_image_filename.extend([os.path.join(folder, name) for name in os.listdir(folder)])

        random.shuffle(train_image_filename)
        random.shuffle(val_image_filename)
        random.shuffle(test_image_filename)

        return folders_train, folders_val, folders_test, train_image_filename, val_image_filename, test_image_filename

    pass


class MiniImageNetIC(Dataset):

    def __init__(self, image_filename_list, transform=None):
        self.transform = transform
        self.image_filename_list = image_filename_list

        _file_class = [os.path.basename(os.path.split(file_name)[0]) for file_name in self.image_filename_list]
        _class_name = sorted(set(_file_class))
        self.train_label = [_class_name.index(file_class_name) for file_class_name in _file_class]
        pass

    def __len__(self):
        return len(self.image_filename_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_filename_list[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.train_label[idx]
        return image, label, idx

    @staticmethod
    def get_data_loader(image_filename_list, batch_size=16, shuffle=False):
        transform_test = transforms.Compose([lambda x: np.asarray(x), transforms.ToTensor(),
                                             transforms.Normalize(mean=Config.MEAN_PIXEL, std=Config.STD_PIXEL)])
        dataset = MiniImageNetIC(image_filename_list, transform=transform_test)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    pass


class MiniImageNet(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.ic_ids = self.task.ic_train_ids if self.split == 'train' else self.task.ic_test_ids
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image = Image.open(self.image_roots[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)

        ic_id = self.ic_ids[idx] if self.ic_ids else -1
        return image, label, ic_id

    @staticmethod
    def get_data_loader(task, num_per_class=1, split='train', sampler_test=False, shuffle=False):
        normalize = transforms.Normalize(mean=Config.MEAN_PIXEL, std=Config.STD_PIXEL)
        transform_train = transforms.Compose([transforms.RandomCrop(84, padding=8), transforms.RandomHorizontalFlip(),
                                              lambda x: np.asarray(x), transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([lambda x: np.asarray(x), transforms.ToTensor(), normalize])

        if split == 'train':
            dataset = MiniImageNet(task, split=split, transform=transform_train)
            sampler = ClassBalancedSampler(num_per_class, task.num_way, task.num_shot, shuffle=shuffle)
        else:
            dataset = MiniImageNet(task, split=split, transform=transform_test)
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_way, task.batch_size, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_way, task.batch_size, shuffle=shuffle)
                pass
            pass

        return DataLoader(dataset, batch_size=num_per_class * task.num_way, sampler=sampler)

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
    def knn(epoch, feature_encoder, ic_model, low_dim, train_loader, k, t=0.1):
        with torch.no_grad():

            def _cal(_labels, _dist, _train_labels, _retrieval_1_hot, _top1, _top5, _max_c):
                # ---------------------------------------------------------------------------------- #
                _batch_size = _labels.size(0)
                _yd, _yi = _dist.topk(k+1, dim=1, largest=True, sorted=True)
                _yd, _yi = _yd[:, 1:], _yi[:, 1:]
                _candidates = train_labels.view(1, -1).expand(_batch_size, -1)
                _retrieval = torch.gather(_candidates, 1, _yi)

                _retrieval_1_hot.resize_(_batch_size * k, _max_c).zero_()
                _retrieval_1_hot = _retrieval_1_hot.scatter(1, _retrieval.view(-1, 1), 1).view(_batch_size, -1, _max_c)
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

            # Tools.print("Test 1 {} Top1={:.2f} Top5={:.2f}".format(epoch, top1 * 100. / total, top5 * 100. / total))
            return top1 / total, top5 / total

        pass

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # model
        self.feature_encoder = cuda(CNNEncoder())
        self.relation_network = cuda(RelationNetwork())
        self.ic_model = cuda(ICModel(in_dim=Config.ic_in_dim, out_dim=Config.ic_out_dim))

        # optim
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=Config.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, Config.train_episode//3, gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=Config.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, Config.train_episode//3, gamma=0.5)
        self.ic_model_optim = torch.optim.Adam(self.ic_model.parameters(), lr=Config.learning_rate)
        self.ic_model_scheduler = StepLR(self.ic_model_optim, Config.train_episode//3, gamma=0.5)

        # all data, 后三个用于测试IC的性能
        (self.folders_train, self.folders_val, self.folders_test, self.train_image_filename_list,
         self.val_image_filename_list, self.test_image_filename_list) = MiniImageNetTask.folders(Config.data_root)
        # 用于测试IC的性能
        self.ic_test_train_loader = MiniImageNetIC.get_data_loader(self.train_image_filename_list, 32, shuffle=False)
        self.ic_test_val_loader = MiniImageNetIC.get_data_loader(self.val_image_filename_list, 32, shuffle=False)
        self.ic_test_test_loader = MiniImageNetIC.get_data_loader(self.test_image_filename_list, 32, shuffle=False)

        # DHC
        self.produce_class1 = ProduceClass(len(self.train_image_filename_list), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class2 = ProduceClass(len(self.train_image_filename_list), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class1.init()
        self.produce_class2.init()

        self.ic_loss = cuda(nn.CrossEntropyLoss())
        self.fsl_loss = cuda(nn.MSELoss())
        self.fsl_ic_loss = cuda(nn.MSELoss())
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

        self.produce_class1.reset()
        all_loss, all_loss_fsl, all_loss_fsl_ic, all_loss_ic_sample, all_loss_ic_batch = 0.0, 0.0, 0.0, 0.0, 0.0
        for episode in range(Config.train_episode):
            self.feature_encoder.train()
            self.relation_network.train()
            self.ic_model.train()

            ###########################################################################
            # 1 init task
            task = MiniImageNetTask(self.folders_train, Config.task_batch_size,
                                    Config.num_way, Config.num_shot, self.train_image_filename_list)
            sample_data_loader = MiniImageNet.get_data_loader(task, Config.num_shot, split="train", shuffle=False)
            batch_data_loader = MiniImageNet.get_data_loader(task, Config.task_batch_size, split="val", shuffle=True)
            samples, sample_labels, ic_sample_ids = sample_data_loader.__iter__().next()
            batches, batch_labels, ic_batch_ids = batch_data_loader.__iter__().next()
            samples, ic_sample_ids = cuda(samples), cuda(ic_sample_ids)
            batches, batch_labels, ic_batch_ids = cuda(batches), cuda(batch_labels), cuda(ic_batch_ids)
            ###########################################################################

            ###########################################################################
            # 2 calculate features
            relations, sample_features, batch_features = self.compare_fsl(samples, batches)
            ic_sample_out_logits, ic_sample_out_l2norm, ic_sample_out_sigmoid = self.ic_model(sample_features)
            ic_batch_out_logits, ic_batch_out_l2norm, ic_batch_out_sigmoid = self.ic_model(batch_features)
            ###########################################################################

            ###########################################################################
            self.produce_class1.cal_label(ic_sample_out_l2norm, ic_sample_ids)
            self.produce_class1.cal_label(ic_batch_out_l2norm, ic_batch_ids)
            ic_sample_targets = self.produce_class2.get_label(ic_sample_ids)
            ic_batch_targets = self.produce_class2.get_label(ic_batch_ids)
            ###########################################################################

            ###########################################################################
            # 3 loss
            ic_sample_mse_sigmoid = torch.index_select(ic_sample_out_sigmoid, 0, batch_labels)
            one_hot_labels = cuda(torch.zeros(Config.task_batch_size * Config.num_way,
                                              Config.num_way)).scatter(1, batch_labels.long().view(-1, 1), 1)

            loss_fsl = self.fsl_loss(relations, one_hot_labels) * 1.0  # a
            loss_fsl_ic = self.fsl_ic_loss(ic_batch_out_sigmoid, ic_sample_mse_sigmoid) * 1.0
            loss_ic_sample = self.ic_loss(ic_sample_out_logits, ic_sample_targets) * 1.0
            loss_ic_batch = self.ic_loss(ic_batch_out_logits, ic_batch_targets) * 1.0
            loss = loss_fsl + loss_fsl_ic + loss_ic_sample + loss_ic_batch

            all_loss += loss.item()
            all_loss_fsl += loss_fsl.item()
            all_loss_fsl_ic += loss_fsl_ic.item()
            all_loss_ic_sample += loss_ic_sample.item()
            all_loss_ic_batch += loss_ic_batch.item()
            ###########################################################################

            ###########################################################################
            # 4 backward
            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()
            self.ic_model.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
            self.feature_encoder_optim.step()
            self.feature_encoder_scheduler.step()
            torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)
            self.relation_network_optim.step()
            self.relation_network_scheduler.step()
            torch.nn.utils.clip_grad_norm_(self.ic_model.parameters(), 0.5)
            self.ic_model_optim.step()
            self.ic_model_scheduler.step()
            ###########################################################################

            ###########################################################################
            # val ic
            if episode % Config.val_ic_freq == 0 and episode > 0:
                Tools.print()
                Tools.print("Test IC {} .......".format(episode))
                self.feature_encoder.eval()
                self.ic_model.eval()
                self.val_ic(episode, ic_loader=self.ic_test_train_loader, name="Train")
                self.val_ic(episode, ic_loader=self.ic_test_val_loader, name="Val")
                self.val_ic(episode, ic_loader=self.ic_test_test_loader, name="Test")

                # 切换
                classes = self.produce_class2.classes
                self.produce_class2.classes = self.produce_class1.classes
                self.produce_class1.classes = classes
                Tools.print("Train: [{}] {}/{}".format(
                    episode, self.produce_class1.count, self.produce_class1.count_2))
                self.produce_class1.reset()
                Tools.print()
                pass

            # val fsl
            if episode % Config.val_fsl_freq == 0 and episode > 0 :
                Tools.print()
                Tools.print("Val...")
                self.val_fsl_train(episode)
                val_accuracy = self.val_fsl_val(episode)

                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.feature_encoder.state_dict(), Config.fe_dir)
                    torch.save(self.relation_network.state_dict(), Config.rn_dir)
                    torch.save(self.ic_model.state_dict(), Config.ic_dir)
                    Tools.print("Save networks for episode: {}".format(episode))
                    pass

                all_loss, all_loss_fsl, all_loss_fsl_ic = 0.0, 0.0, 0.0
                all_loss_ic_sample, all_loss_ic_batch, all_loss_ic_batch_2 = 0.0, 0.0, 0.0
                Tools.print()
                pass
            ###########################################################################

            ###########################################################################
            # print
            if episode % Config.print_freq == 0 and episode > 0:
                _e = episode % Config.val_fsl_freq + 1
                Tools.print("{:6} loss:{:.3f}/{:.3f} fsl:{:.3f}/{:.3f} ic-s:{:.3f}/{:.3f} "
                            "ic-b:{:.3f}/{:.3f} fsl-ic:{:.3f}/{:.3f} lr:{}".format(
                    episode + 1, all_loss / _e, loss.item(), all_loss_fsl / _e, loss_fsl.item(),
                    all_loss_ic_sample / _e, loss_ic_sample.item(), all_loss_ic_batch / _e, loss_ic_batch.item(),
                    all_loss_fsl_ic / _e, loss_fsl_ic.item(), self.feature_encoder_scheduler.get_lr()))
                pass
            ###########################################################################
            pass

        pass

    def val_ic(self, episode, ic_loader, name="Test"):
        acc_1, acc_2 = KNN.knn(episode, self.feature_encoder, self.ic_model, Config.ic_out_dim, ic_loader, 100)
        Tools.print("Epoch: [{}] {} {:.4f}/{:.4f}".format(episode, name, acc_1, acc_2))
        pass

    def val_fsl_train(self, episode):
        val_train_accuracy = self._val(self.folders_train, sampler_test=False, all_episode=Config.val_episode)
        Tools.print("Val {} Train Accuracy: {}".format(episode, val_train_accuracy))
        return val_train_accuracy

    def val_fsl_val(self, episode):
        val_accuracy = self._val(self.folders_val, sampler_test=False, all_episode=Config.val_episode)
        Tools.print("Val {} Val Accuracy: {}".format(episode, val_accuracy))
        return val_accuracy

    def val_fsl_test(self, episode):
        Tools.print()
        Tools.print("Testing...")
        total_accuracy = 0.0
        for _which in range(Config.test_avg_num):
            test_accuracy = self._val(self.folders_test, sampler_test=True, all_episode=Config.test_episode)
            total_accuracy += test_accuracy
            Tools.print("Episode={}, {} Test accuracy={}, Total accuracy={}".format(
                episode, _which, test_accuracy, total_accuracy))
            pass

        final_accuracy = total_accuracy / Config.test_avg_num
        Tools.print("Final accuracy: {}".format(final_accuracy))
        return final_accuracy

    def _val(self, folders, sampler_test, all_episode):
        accuracies = []
        for i in range(all_episode):
            total_rewards, counter = 0, 0
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本
            task = MiniImageNetTask(folders, Config.task_batch_size, Config.num_way, Config.num_shot)
            sample_data_loader = MiniImageNet.get_data_loader(task, 1, "train", sampler_test, shuffle=False)
            batch_data_loader = MiniImageNet.get_data_loader(task, 3, "val",  sampler_test=sampler_test, shuffle=True)
            samples, labels, _ = sample_data_loader.__iter__().next()
            samples = cuda(samples)

            for batches, batch_labels, _ in batch_data_loader:
                ###########################################################################
                # calculate features
                batches = cuda(batches)
                relations, _, _ = self.compare_fsl(samples, batches)
                ###########################################################################

                _, predict_labels = torch.max(relations.data, 1)
                batch_size = batch_labels.shape[0]
                rewards = [1 if predict_labels[j].cpu() == batch_labels[j] else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)

                counter += batch_size
                pass

            accuracies.append(total_rewards / 1.0 / counter)
            pass
        return np.mean(np.array(accuracies, dtype=np.float))

    def compare_fsl(self, samples, batches):
        # calculate features
        sample_features = self.feature_encoder(samples)  # 5x64*19*19
        batch_features = self.feature_encoder(batches)  # 75x64*19*19
        batch_size, feature_dim, feature_width, feature_height = batch_features.shape

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(
            Config.num_shot * Config.num_way, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext),
                                   2).view(-1, feature_dim * 2, feature_width, feature_height)

        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, Config.num_way * Config.num_shot)

        return relations, sample_features, batch_features

    pass


##############################################################################################################


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_name = "1"
    train_episode = 300000
    learning_rate = 0.001

    num_way = 5
    num_shot = 1
    task_batch_size = 15
    val_episode = 600
    test_avg_num = 2
    test_episode = 600

    print_freq = 1000
    val_ic_freq = 1000
    val_fsl_freq = 5000  # 5000

    # ic
    ic_in_dim = 64
    ic_out_dim = 200
    ic_ratio = 3

    MEAN_PIXEL = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    STD_PIXEL = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    fe_dir = Tools.new_dir("../models/ic_fsl_old/{}_fe_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    rn_dir = Tools.new_dir("../models/ic_fsl_old/{}_rn_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    ic_dir = Tools.new_dir("../models/ic_fsl_old/{}_ic_{}way_{}shot.pkl".format(model_name, num_way, num_shot))
    pass


"""
# 0.7015 / 0.5001
# 0.7038 / 0.5209
# 0.6790 / 0.5189
# 0.6919 / 0.5198  # small net, fsl+ic-s+ic-b+fsl-ic(cross)
# 0.7100 / 0.5278  # large net, fsl+ic-s+ic-b
"""

if __name__ == '__main__':
    runner = Runner()
    runner.load_model()

    runner.val_fsl_train(episode=0)
    runner.val_fsl_val(episode=0)
    runner.val_fsl_test(episode=0)

    runner.train()

    runner.load_model()
    runner.val_fsl_train(episode=Config.train_episode)
    runner.val_fsl_val(episode=Config.train_episode)
    runner.val_fsl_test(episode=Config.train_episode)
    pass
