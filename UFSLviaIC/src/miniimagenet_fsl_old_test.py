import os
import math
import torch
import random
import platform
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.utils as vutils
from alisuretool.Tools import Tools
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
    def get_data_loader(task, num_per_class=1, split='train', sampler_test=False, shuffle=False):
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)), transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #     transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        # transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if split == 'train':
            dataset = MiniImageNet(task, split=split, transform=transform_train)
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            dataset = MiniImageNet(task, split=split, transform=transform_test)
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

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


class CNNEncoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
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


class RelationNetwork1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 5 * 5, 64)  # 64
        self.fc2 = nn.Linear(64, 1)  # 64
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

##############################################################################################################


class Runner(object):

    def __init__(self):
        self.feature_encoder = cuda(Config.feature_encoder)
        self.relation_network = cuda(Config.relation_network)

        # data
        self.folders_train, self.folders_val, self.folders_test = MiniImageNet.folders(Config.data_root)
        pass

    def load_model(self):
        if os.path.exists(Config.fe_dir):
            self.feature_encoder.load_state_dict(torch.load(Config.fe_dir))
            Tools.print("load feature encoder success from {}".format(Config.fe_dir))

        if os.path.exists(Config.rn_dir):
            self.relation_network.load_state_dict(torch.load(Config.rn_dir))
            Tools.print("load relation network success from {}".format(Config.rn_dir))
        pass

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
        relation_pairs = torch.cat((sample_features_ext,
                                    batch_features_ext),2).view(-1, feature_dim * 2, feature_width, feature_height)

        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, Config.num_way * Config.num_shot)
        return relations

    def val_train(self, episode):
        val_train_accuracy = self._val(self.folders_train, sampler_test=False, all_episode=Config.test_episode)
        Tools.print("Val {} Train Accuracy: {}".format(episode, val_train_accuracy))
        return val_train_accuracy

    def val_val(self, episode):
        val_accuracy = self._val(self.folders_val, sampler_test=False, all_episode=Config.test_episode)
        Tools.print("Val {} Accuracy: {}".format(episode, val_accuracy))
        return val_accuracy

    def val_test(self):
        Tools.print()
        Tools.print("Testing...")
        total_accuracy = 0.0
        for episode in range(Config.test_avg_num):
            test_accuracy = self._val(self.folders_test, sampler_test=True, all_episode=Config.test_episode)
            total_accuracy += test_accuracy
            Tools.print("episode={}, Test accuracy={}, Total accuracy={}".format(
                episode, test_accuracy, total_accuracy))
            pass

        final_accuracy = total_accuracy / Config.test_avg_num
        Tools.print("Final accuracy: {}".format(final_accuracy))
        return final_accuracy

    def _val(self, folders, sampler_test, all_episode):
        accuracies = []
        for i in range(all_episode):
            total_rewards = 0
            counter = 0
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本
            task = MiniImageNetTask(folders, Config.num_way, Config.num_shot, Config.task_batch_size)
            sample_data_loader = MiniImageNet.get_data_loader(task, 1, "train", sampler_test=sampler_test, shuffle=False)
            batch_data_loader = MiniImageNet.get_data_loader(task, 3, "val", sampler_test=sampler_test, shuffle=True)
            samples, labels = sample_data_loader.__iter__().next()
            samples = cuda(samples)

            for batches, batch_labels in batch_data_loader:
                ###########################################################################
                # calculate features
                batches = cuda(batches)
                relations= self.compare_fsl(samples, batches)
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

    pass


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_way = 5
    num_shot = 1
    task_batch_size = 15

    test_avg_num = 5
    test_episode = 600

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    # model_path = "fsl"
    # model_fe_name = "1_64_5_1_fe_5way_1shot.pkl"
    # model_rn_name = "1_64_5_1_rn_5way_1shot.pkl"
    # feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
    # fe_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_fe_name))
    # rn_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_rn_name))

    # model_path = "ic_fsl"
    # model_fe_name = "2_64_5_1_fe_5way_1shot.pkl"
    # model_rn_name = "2_64_5_1_rn_5way_1shot.pkl"
    # feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
    # fe_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_fe_name))
    # rn_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_rn_name))

    # model_path = "train_one_shot_alisure"
    # model_fe_name = "1fe_5way_1shot.pkl"
    # model_rn_name = "1_rn_5way_1shot.pkl"
    # feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
    # fe_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_fe_name))
    # rn_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_rn_name))

    # model_path = "train_one_shot_alisure"
    # model_fe_name = "2_fe_5way_1shot.pkl"
    # model_rn_name = "2_rn_5way_1shot.pkl"
    # feature_encoder, relation_network = CNNEncoder1(), RelationNetwork1()
    # fe_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_fe_name))
    # rn_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_rn_name))

    # model_path = "ic_ufsl"
    # model_fe_name = "2_64_5_1_fe_5way_1shot.pkl"
    # model_rn_name = "2_64_5_1_rn_5way_1shot.pkl"
    # feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
    # fe_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_fe_name))
    # rn_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_rn_name))

    # model_path = "fsl2"
    # model_fe_name = "1_64_5_1_fe_5way_1shot.pkl"
    # model_rn_name = "1_64_5_1_rn_5way_1shot.pkl"
    # feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
    # fe_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_fe_name))
    # rn_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_rn_name))

    model_path = "fsl2"
    model_fe_name = "1_64_5_1_fe_5way_1shot.pkl"
    model_rn_name = "1_64_5_1_rn_5way_1shot.pkl"
    feature_encoder, relation_network = CNNEncoder(), RelationNetwork()
    fe_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_fe_name))
    rn_dir = Tools.new_dir("../models/{}/{}".format(model_path, model_rn_name))
    pass


##############################################################################################################


if __name__ == '__main__':
    runner = Runner()

    runner.load_model()

    runner.val_train(0)
    runner.val_val(0)
    runner.val_test()
    pass
