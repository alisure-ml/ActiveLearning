import os
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset


##############################################################################################################


class ClassBalancedSampler(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
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

    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = dict(zip(class_folders, np.array(range(len(class_folders)))))

        samples = dict()
        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num + test_num]
            pass

        self.train_labels = [labels[os.path.split(x)[0]] for x in self.train_roots]
        self.test_labels = [labels[os.path.split(x)[0]] for x in self.test_roots]
        pass

    pass


class MiniImageNet(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
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
            # normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            pass
        dataset = MiniImageNet(task, split=split, transform=transform)
        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

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

    def val_train(self):
        return self._val(self.folders_train, sampler_test=False, all_episode=self.test_episode)

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

        mean_acc = np.mean(acc_list, axis=0)
        if is_print:
            for acc in acc_list:
                Tools.print("episode={}, Test accuracy={}".format(episode, acc))
                pass
            Tools.print("episode={}, Mean Test accuracy={}".format(episode, mean_acc))
            pass
        return mean_acc

    def val(self, episode=0, is_print=True):
        acc_train = self.val_train()
        acc_val = self.val_val()
        acc_test1 = self.val_test()
        acc_test2 = self.val_test2()

        if is_print:
            Tools.print("Train {} Accuracy: {}".format(episode, acc_train))
            Tools.print("Val   {} Accuracy: {}".format(episode, acc_val))
            Tools.print("Test1 {} Accuracy: {}".format(episode, acc_test1))
            Tools.print("Test2 {} Accuracy: {}".format(episode, acc_test2))
            pass
        return acc_val

    def _val(self, folders, sampler_test, all_episode):
        accuracies = []
        for i in range(all_episode):
            total_rewards_1, total_rewards_2, counter = 0, 0, 0
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
                batch_size = batch_labels.shape[0]
                counter += batch_size

                relations = self.model_fn(samples, self.to_cuda(batches))

                # method 1
                relations_sum = np.zeros((batch_size, self.num_way))
                for _i1, relation in enumerate(relations.data):
                    for _i2, r in enumerate(relation):
                        relations_sum[_i1][labels[_i2]] += r
                    pass
                final_predict_labels_1 = list(np.argmax(relations_sum, -1))
                rewards_1 = [1 if final_predict_labels_1[j] == batch_labels[j] else 0 for j in range(batch_size)]
                total_rewards_1 += np.sum(rewards_1)

                # method 2
                if self.num_shot > 1:
                    _, predict_labels = torch.max(relations.data, 1)
                    final_predict_labels_2 = [int(labels[predict]) for predict in predict_labels]
                    rewards_2 = [1 if final_predict_labels_2[j] == batch_labels[j] else 0 for j in range(batch_size)]
                    total_rewards_2 += np.sum(rewards_2)
                pass
            if self.num_shot > 1:
                accuracies.append([total_rewards_1 / 1.0 / counter, total_rewards_2 / 1.0 / counter])
            else:
                accuracies.append(total_rewards_1 / 1.0 / counter)
            pass
        return np.mean(np.array(accuracies, dtype=np.float), axis=0)

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    pass


if __name__ == '__main__':
    test_tool = TestTool(model_fn=None, data_root=None, num_way=5, num_shot=1, episode_size=15, test_episode=600)
    _acc = test_tool.val(episode=0, is_print=True)
    test_tool.test(5, episode=0, is_print=True)
    pass
