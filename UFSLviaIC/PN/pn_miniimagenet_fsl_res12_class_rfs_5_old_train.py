import os
import time
import torch
import pickle
import random
import platform
import warnings
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
from alisuretool.Tools import Tools
from torch.distributions import Bernoulli
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from pn_miniimagenet_fsl_test_tool import ClassBalancedSampler, ClassBalancedSamplerTest, AverageMeter


##############################################################################################################


class MiniImageNetTask(object):

    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(character_folders, self.num_classes)
        labels = dict(zip(class_folders, np.array(range(len(class_folders)))))

        self.train_datas, self.train_labels = [], []
        self.test_datas, self.test_labels = [], []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            sample_id_list = random.sample(range(len(temp)), train_num + test_num)
            train_id = sample_id_list[:train_num]
            test_id = sample_id_list[train_num:train_num + test_num]

            self.train_datas += [temp[sample_id] for sample_id in train_id]
            self.train_labels += [labels[c] for _ in train_id]
            self.test_datas += [temp[sample_id] for sample_id in test_id]
            self.test_labels += [labels[c] for _ in test_id]
            pass
        pass

    pass


class MiniImageNet(Dataset):

    def __init__(self, task, split='train', transform=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_datas if self.split == 'train' else self.task.test_datas
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image = self.image_roots[idx]
        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

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
        return folders_train, folders_val, folders_test

    @staticmethod
    def get_data_loader(task, num_per_class=1, split='train', sampler_test=False, shuffle=False, transform=None):
        if split == 'train':
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        if transform is None:
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                             std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            pass
        dataset = MiniImageNet(task, split=split, transform=transform)
        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    pass


class ImageNet(Dataset):

    def __init__(self, data_list, is_train=True):
        super(Dataset, self).__init__()
        self.data_list = data_list
        self.is_train = is_train

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
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        if not is_train:
            self.transform = self.transform_test
        pass

    def __getitem__(self, item):
        now_label_image_tuple = self.data_list[item]
        now_index, now_label, now_image_filename = now_label_image_tuple

        image = Image.open(now_image_filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, now_label, item

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

    pass


class MiniImageNetTaskOld(object):

    def __init__(self, image_data_list, num_classes, train_num, test_num):
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_keys = random.sample(list(image_data_list.keys()), self.num_classes)
        labels = dict(zip(class_keys, np.array(range(len(class_keys)))))

        self.train_datas, self.train_labels = [], []
        self.test_datas, self.test_labels = [], []
        for c in class_keys:
            temp = image_data_list[c]
            sample_id_list = random.sample(range(len(temp)), train_num + test_num)
            train_id = sample_id_list[:train_num]
            test_id = sample_id_list[train_num:train_num + test_num]

            self.train_datas += [temp[sample_id] for sample_id in train_id]
            self.train_labels += [labels[c] for _ in train_id]
            self.test_datas += [temp[sample_id] for sample_id in test_id]
            self.test_labels += [labels[c] for _ in test_id]
            pass
        pass

    pass


class MiniImageNetOld(Dataset):

    def __init__(self, task, split='train', transform=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_datas = self.task.train_datas if self.split == 'train' else self.task.test_datas
        pass

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    @staticmethod
    def all_data(data_root):
        test_pickle = os.path.join(data_root, 'miniImageNet_category_split_test.pickle')
        val_pickle = os.path.join(data_root, 'miniImageNet_category_split_val.pickle')

        with open(test_pickle, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            test_imgs = data['data']
            test_labels = data['labels']
            pass

        test_data = {}
        for idx in range(test_imgs.shape[0]):
            if test_labels[idx] not in test_data:
                test_data[test_labels[idx]] = []
            test_data[test_labels[idx]].append(test_imgs[idx])
            pass

        with open(val_pickle, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            val_imgs = data['data']
            val_labels = data['labels']
            pass

        val_data = {}
        for idx in range(val_imgs.shape[0]):
            if val_labels[idx] not in val_data:
                val_data[val_labels[idx]] = []
            val_data[val_labels[idx]].append(val_imgs[idx])
            pass

        return val_data, test_data

    @staticmethod
    def get_data_loader(task, num_per_class=1, split='train', sampler_test=False, shuffle=False, transform=None):
        if split == 'train':
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        if transform is None:
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                             std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            pass
        dataset = MiniImageNetOld(task, split=split, transform=transform)
        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    pass


class ImageNetOld(Dataset):
    def __init__(self, imgs, labels, is_train=True):
        super(Dataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.is_train = is_train

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                         std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([lambda x: Image.fromarray(x), transforms.ToTensor(), normalize])
        if not is_train:
            self.transform = self.transform_test
        pass

    @staticmethod
    def get_all_data(data_root):
        file_train = os.path.join(data_root, 'miniImageNet_category_split_train_phase_train.pickle')
        with open(file_train, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            train_imgs = data['data']
            train_labels = data['labels']
            pass

        file_val = os.path.join(data_root, 'miniImageNet_category_split_train_phase_val.pickle')
        with open(file_val, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            val_imgs = data['data']
            val_labels = data['labels']
            pass
        return train_imgs, train_labels, val_imgs, val_labels

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)
        return img, target, item

    def __len__(self):
        return len(self.labels)

    pass


##############################################################################################################


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class DropBlock(nn.Module):

    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask

    pass


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        pass

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

    pass


class ResNet(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, num_classes=-1):
        super(ResNet, self).__init__()
        self.inplanes = 3
        self.layer1 = self._make_layer(block, n_blocks[0], 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320, stride=2,
                                       drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640, stride=2,
                                       drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block, block_size=block_size)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x
        if self.num_classes > 0:
            x = self.classifier(x)
        return feat, x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


##############################################################################################################


class TestTool(object):

    def __init__(self, model_fn, data_root, num_way=5, num_shot=1, episode_size=15, test_episode=600, transform=None):
        self.model_fn = model_fn
        self.transform = transform

        self.folders_train, self.folders_val, self.folders_test = MiniImageNet.folders(data_root)

        self.test_episode = test_episode
        self.num_way = num_way
        self.num_shot = num_shot
        self.episode_size = episode_size
        pass

    def val_val(self):
        return self._val(self.folders_val, sampler_test=False, all_episode=self.test_episode)

    def val_test(self):
        return self._val(self.folders_test, sampler_test=False, all_episode=self.test_episode)

    def val_test2(self):
        return self._val(self.folders_test, sampler_test=True, all_episode=self.test_episode)

    def test(self, test_avg_num, episode=0, is_print=True):
        acc_list = []
        for _ in range(test_avg_num):
            acc = self._val(self.folders_test, sampler_test=True, all_episode=self.test_episode)
            acc_list.append(acc)
            pass

        mean_acc = np.mean(acc_list)
        if is_print:
            for acc in acc_list:
                Tools.print("episode={}, Test accuracy={}".format(episode, acc))
                pass
            Tools.print("episode={}, Mean Test accuracy={}".format(episode, mean_acc))
            pass
        return mean_acc

    def val(self, episode=0, is_print=True):
        acc_val = self.val_val()
        acc_test1 = self.val_test()
        acc_test2 = self.val_test2()

        if is_print:
            Tools.print("Val   {} Accuracy: {}".format(episode, acc_val))
            Tools.print("Test1 {} Accuracy: {}".format(episode, acc_test1))
            Tools.print("Test2 {} Accuracy: {}".format(episode, acc_test2))
            pass
        return acc_val

    def _val(self, folders, sampler_test, all_episode):
        accuracies = []
        for i in range(all_episode):
            total_rewards, counter = 0, 0
            # 随机选5类，每类中取出num_shot个作为训练样本，总共取出15个作为测试样本
            task = MiniImageNetTask(folders, self.num_way, self.num_shot, self.episode_size)
            sample_data_loader = MiniImageNet.get_data_loader(task, self.num_shot, "train", sampler_test=sampler_test,
                                                              shuffle=False, transform=self.transform)
            num_per_class = 5 if self.num_shot > 1 else 3
            batch_data_loader = MiniImageNet.get_data_loader(task, num_per_class, "val", sampler_test=sampler_test,
                                                             shuffle=True, transform=self.transform)
            samples, labels = sample_data_loader.__iter__().next()

            samples = self.to_cuda(samples)
            for batches, batch_labels in batch_data_loader:
                results = self.model_fn(samples, self.to_cuda(batches), num_way=self.num_way, num_shot=self.num_shot)

                _, predict_labels = torch.max(results.data, 1)
                batch_size = batch_labels.shape[0]
                rewards = [1 if predict_labels[j].cpu() == batch_labels[j] else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)
                counter += batch_size
                pass
            accuracies.append(total_rewards / 1.0 / counter)
            pass
        return np.mean(np.array(accuracies, dtype=np.float))

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    pass


class TestToolOld(object):

    def __init__(self, model_fn, data_root, num_way=5, num_shot=1, episode_size=15, test_episode=600, transform=None):
        self.model_fn = model_fn
        self.transform = transform

        self.val_data, self.test_data = MiniImageNetOld.all_data(data_root)

        self.test_episode = test_episode
        self.num_way = num_way
        self.num_shot = num_shot
        self.episode_size = episode_size
        pass

    def val_val(self):
        return self._val(self.val_data, sampler_test=False, all_episode=self.test_episode)

    def val_test(self):
        return self._val(self.test_data, sampler_test=False, all_episode=self.test_episode)

    def val_test2(self):
        return self._val(self.test_data, sampler_test=True, all_episode=self.test_episode)

    def test(self, test_avg_num, episode=0, is_print=True):
        acc_list = []
        for _ in range(test_avg_num):
            acc = self._val(self.test_data, sampler_test=True, all_episode=self.test_episode)
            acc_list.append(acc)
            pass

        mean_acc = np.mean(acc_list)
        if is_print:
            for acc in acc_list:
                Tools.print("episode={}, Test accuracy={}".format(episode, acc))
                pass
            Tools.print("episode={}, Mean Test accuracy={}".format(episode, mean_acc))
            pass
        return mean_acc

    def val(self, episode=0, is_print=True):
        acc_val = self.val_val()
        acc_test1 = self.val_test()
        acc_test2 = self.val_test2()

        if is_print:
            Tools.print("Val   {} Accuracy: {}".format(episode, acc_val))
            Tools.print("Test1 {} Accuracy: {}".format(episode, acc_test1))
            Tools.print("Test2 {} Accuracy: {}".format(episode, acc_test2))
            pass
        return acc_val

    def _val(self, test_data, sampler_test, all_episode):
        accuracies = []
        for i in range(all_episode):
            total_rewards, counter = 0, 0
            # 随机选5类，每类中取出num_shot个作为训练样本，总共取出15个作为测试样本
            task = MiniImageNetTaskOld(test_data, self.num_way, self.num_shot, self.episode_size)
            sample_data_loader = MiniImageNetOld.get_data_loader(
                task, self.num_shot, "train", sampler_test=sampler_test, shuffle=False, transform=self.transform)
            num_per_class = 5 if self.num_shot > 1 else 3
            batch_data_loader = MiniImageNetOld.get_data_loader(
                task, num_per_class, "val", sampler_test=sampler_test, shuffle=True, transform=self.transform)
            samples, labels = sample_data_loader.__iter__().next()

            samples = self.to_cuda(samples)
            for batches, batch_labels in batch_data_loader:
                results = self.model_fn(samples, self.to_cuda(batches), num_way=self.num_way, num_shot=self.num_shot)

                _, predict_labels = torch.max(results.data, 1)
                batch_size = batch_labels.shape[0]
                rewards = [1 if predict_labels[j].cpu() == batch_labels[j] else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)
                counter += batch_size
                pass
            accuracies.append(total_rewards / 1.0 / counter)
            pass
        return np.mean(np.array(accuracies, dtype=np.float))

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    pass


class Runner(object):

    def __init__(self):
        # model
        self.model = ResNet(BasicBlock, [1, 1, 1, 1], avg_pool=True,
                            drop_rate=0.1, dropblock_size=5, num_classes=64).cuda()

        # all data
        if Config.class_is_old:
            train_imgs, train_labels, val_imgs, val_labels = ImageNetOld.get_all_data(Config.data_root2)
            self.train_loader = DataLoader(ImageNetOld(imgs=train_imgs, labels=train_labels, is_train=True),
                                           batch_size=Config.batch_size, shuffle=True,
                                           drop_last=True, num_workers=Config.num_workers)
            self.val_loader = DataLoader(ImageNetOld(imgs=val_imgs, labels=val_labels, is_train=False),
                                         batch_size=Config.batch_size, shuffle=False,
                                         drop_last=False, num_workers=Config.num_workers)

            # test tool
            self.test_tool = TestToolOld(self.proto_test, data_root=Config.data_root2,
                                         num_way=Config.num_way, num_shot=Config.num_shot,
                                         episode_size=Config.episode_size, test_episode=Config.test_episode)
        else:
            data_train_list, data_val_list, data_test_list = ImageNet.get_data_all(Config.data_root)
            self.train_loader = DataLoader(ImageNet(data_list=data_train_list, is_train=True),
                                           batch_size=Config.batch_size, shuffle=True,
                                           drop_last=True, num_workers=Config.num_workers)
            self.val_loader = DataLoader(ImageNet(data_list=data_val_list, is_train=False),
                                         batch_size=Config.batch_size, shuffle=False,
                                         drop_last=False, num_workers=Config.num_workers)

            # test tool
            self.test_tool = TestTool(self.proto_test, data_root=Config.data_root,
                                      num_way=Config.num_way, num_shot=Config.num_shot,
                                      episode_size=Config.episode_size, test_episode=Config.test_episode)
            pass

        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss().cuda()
        pass

    def load_model(self):
        model_path = os.path.join(Config.model_path, 'last.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            Tools.print("load model success from {}".format(model_path))
            pass
        pass

    @staticmethod
    def normalize(x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

    def proto_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        _, sample_z = self.model(samples)  # 5x64*5*5
        _, batch_z = self.model(batches)  # 75x64*5*5
        sample_z = self.normalize(sample_z)
        batch_z = self.normalize(batch_z)

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
        for epoch in range(1, Config.epochs + 1):
            self.adjust_learning_rate(epoch)
            Tools.print()
            Tools.print("==> Training ... {} lr={}".format(epoch, self.optimizer.param_groups[0]["lr"]))

            time1 = time.time()
            train_acc, train_acc_top5, train_loss = self.train_epoch(epoch)
            time2 = time.time()
            Tools.print('Epoch {}, {:.4f} {:.4f} {:.4f}, total time {:.2f}'.format(
                epoch, train_acc, train_acc_top5, train_loss, time2 - time1))

            test_acc, test_acc_top5, test_loss = self.validate()
            Tools.print('Epoch {}, {:.4f} {:.4f} {:.4f}'.format(epoch, test_acc, test_acc_top5, test_loss))

            # regular saving
            if epoch % Config.save_freq == 0:
                state = {'epoch': epoch, 'model': self.model.state_dict()}
                save_file = os.path.join(Config.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

                Tools.print()
                Tools.print('==> Saving ... {}'.format(save_file))

                self.model.eval()
                self.test_tool.val(episode=epoch, is_print=True)
                pass
            pass

        Tools.print()
        torch.save(self.model.state_dict(), os.path.join(Config.model_path, 'last.pth'))
        self.model.eval()
        self.test_tool.val(episode=Config.epochs, is_print=True)
        pass

    def train_epoch(self, epoch):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for idx, (input, target, _) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            input = input.float().cuda()
            target = target.cuda()

            # ===================forward=====================
            _, output = self.model(input)
            loss = self.criterion(output, target)

            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # ===================backward=====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if idx % Config.print_freq == 0:
                Tools.print('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                pass
            pass
        return top1.avg, top5.avg, losses.avg

    def validate(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for idx, (input, target, _) in enumerate(self.val_loader):
                input = input.float().cuda()
                target = target.cuda()

                # compute output
                _, output = self.model(input)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % Config.print_freq == 0:
                    Tools.print('Test: [{0}/{1}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        idx, len(self.val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))
                pass
            pass

        return top1.avg, top5.avg, losses.avg

    def adjust_learning_rate(self, epoch):
        steps = np.sum(epoch > np.asarray(Config.lr_decay_epochs))
        if steps > 0:
            new_lr = Config.learning_rate * (Config.lr_decay_rate ** steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            pass
        pass

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        pass

    pass


##############################################################################################################


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore')

    class_is_old = True

    num_workers = 8
    epochs = 100
    print_freq = 100
    save_freq = 5
    learning_rate = 0.05
    lr_decay_epochs = [60, 80]
    lr_decay_rate = 0.1

    test_episode = 600
    batch_size = 64
    num_way = 5
    num_shot = 1
    episode_size = 15

    model_name = "{}".format('Proto')
    model_path = Tools.new_dir('../../rfs/models_pretrained_my/{}'.format(model_name))

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        data_root2 = '/mnt/4T/Data/data/miniImagenet/miniImageNet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
            data_root2 = '/media/ubuntu/4T/ALISURE/Data/miniImagenet/miniImageNet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pass


if __name__ == '__main__':
    runner = Runner()
    runner.train()

    runner.load_model()
    runner.model.eval()
    runner.test_tool.val(episode=0, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=0, is_print=True)
    pass
