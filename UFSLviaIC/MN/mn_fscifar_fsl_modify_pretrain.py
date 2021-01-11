import os
import sys
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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from mn_tool_fsl_test import FSLTestTool
sys.path.append("../IC")
from fscifar_ic_res import ICResNet
from torchvision.models import resnet18, resnet34, resnet50, vgg16_bn
from mn_tool_net import MatchingNet, Normalize, RunnerTool, ResNet12Small


##############################################################################################################


class CIFARDataset(object):

    def __init__(self, data_list, num_way, num_shot, image_size=32):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        # mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean, std)
        change = transforms.Resize(image_size) if image_size > 32 else lambda x: x

        self.transform = transforms.Compose([
            change, transforms.RandomCrop(image_size, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),

            # change, transforms.RandomResizedCrop(size=image_size),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),

            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([change, transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

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
        task_data = torch.cat([torch.unsqueeze(self.read_image(one[2], self.transform), dim=0) for one in task_list])
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


class Runner(object):

    def __init__(self):
        # all data
        self.data_train = CIFARDataset.get_data_all(Config.data_root)
        self.task_train = CIFARDataset(self.data_train, Config.num_way, Config.num_shot, image_size=Config.image_size)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        self.norm = Normalize(2)

        self.test_tool = FSLTestTool(self.matching_test, data_root=Config.data_root,
                                     num_way=Config.num_way, num_shot=Config.num_shot,
                                     episode_size=Config.episode_size, test_episode=Config.test_episode,
                                     transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load proto net success from {}".format(Config.mn_dir), txt_path=Config.log_file)
        pass

    def matching_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)[0]  # 5x64*5*5
        batch_z = self.matching_net(batches)[0]  # 75x64*5*5
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

    pass


##############################################################################################################


class Config(object):
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    learning_rate = 0.001
    num_workers = 16
    train_epoch = 400

    num_way = 5
    num_shot = 1

    val_freq = 10
    episode_size = 15
    test_episode = 600

    ##############################################################################################################
    # dataset_name = "CIFARFS"
    dataset_name = "FC100"

    image_size = 32
    mn_dir = "../models_CIFARFS/models/ic_res_xx/1_FC100_32_resnet_34_64_512_1_1500_300_200_True_ic.pkl"
    matching_net, net_name, batch_size = ICResNet(low_dim=512, modify_head=True, resnet=resnet34), "conv4", 64
    ##############################################################################################################

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/{}'.format(dataset_name)
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/{}'.format(dataset_name)
    else:
        data_root = "F:\\data\\{}".format(dataset_name)

    log_file = None
    Tools.print(data_root, txt_path=log_file)
    Tools.print(mn_dir, txt_path=log_file)
    pass


##############################################################################################################


"""
2021-01-11 14:43:18 load proto net success from ../models_CIFARFS/models/ic_res_xx/1_FC100_32_resnet_34_64_512_1_1500_300_200_True_ic.pkl
2021-01-11 14:44:00 Train 400 Accuracy: 0.5477777777777778
2021-01-11 14:44:43 Val   400 Accuracy: 0.3092222222222223
2021-01-11 14:45:25 Test1 400 Accuracy: 0.3275555555555556
2021-01-11 14:48:23 Test2 400 Accuracy: 0.3260444444444444
2021-01-11 15:03:19 episode=400, Test accuracy=0.3201777777777778
2021-01-11 15:03:19 episode=400, Test accuracy=0.3203111111111111
2021-01-11 15:03:19 episode=400, Test accuracy=0.3200222222222222
2021-01-11 15:03:19 episode=400, Test accuracy=0.3237555555555556
2021-01-11 15:03:19 episode=400, Test accuracy=0.32106666666666667
2021-01-11 15:03:19 episode=400, Mean Test accuracy=0.32106666666666667
"""


if __name__ == '__main__':
    runner = Runner()
    runner.load_model()
    runner.matching_net.eval()
    runner.test_tool.val(episode=Config.train_epoch, is_print=True, txt_path=Config.log_file)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True, txt_path=Config.log_file)
    pass
