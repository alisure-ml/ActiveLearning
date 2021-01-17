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
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
sys.path.append("../Common")
from UFSLTool import C4Net, Normalize, ResNet12Small


##############################################################################################################


class MiniImageNetIC(Dataset):

    def __init__(self, data_list, image_size=84):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        norm = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                    std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.transform = transforms.Compose([transforms.CenterCrop(size=image_size), transforms.ToTensor(), norm])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image_transform = self.transform(image)
        return image_transform, label, idx

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


##############################################################################################################


class Runner(object):

    def __init__(self, config):
        self.config = config

        # data
        self.data_train, self.data_val, self.data_test = MiniImageNetIC.get_data_all(self.config.data_root)
        self.train_loader = DataLoader(MiniImageNetIC(self.data_train), self.config.batch_size, False, num_workers=self.config.num_workers)
        self.val_loader = DataLoader(MiniImageNetIC(self.data_val), self.config.batch_size, False, num_workers=self.config.num_workers)
        self.test_loader = DataLoader(MiniImageNetIC(self.data_test), self.config.batch_size, False, num_workers=self.config.num_workers)

        # model
        self.net = self.to_cuda(self.config.net)
        pass

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    def load_model(self):
        if os.path.exists(self.config.checkpoint_dir):
            self.net.load_state_dict(torch.load(self.config.checkpoint_dir))
            Tools.print("load mn model success from {}".format(self.config.checkpoint_dir))
        else:
            Tools.print("...................... from {}".format(self.config.checkpoint_dir))
        pass

    def features(self, split="train"):
        Tools.print()
        Tools.print("Vis ...")

        loader = self.test_loader if split == "test" else self.train_loader
        loader = self.val_loader if split == "val" else loader

        feature_list = []
        self.net.eval()
        for image_transform, label, idx in tqdm(loader):
            out = self.net(self.to_cuda(image_transform))
            out = out.view(out.shape[0], -1)
            for i in range(len(idx)):
                feature_list.append([int(idx[i]), int(label[i]), int(label[i]),
                                     np.array(out[i].cpu().detach().numpy()),
                                     np.array(out[i].cpu().detach().numpy())])
                pass
            pass

        Tools.write_to_pkl(os.path.join(self.config.features_dir, "{}.pkl".format(split)), feature_list)
        pass

    pass


##############################################################################################################


class Config(object):

    def __init__(self, checkpoint_dir, features_dir, is_c4net=True):
        self.gpu_id = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.num_workers = 8
        self.batch_size = 8

        self.checkpoint_dir = checkpoint_dir
        self.features_dir = Tools.new_dir(features_dir)

        if is_c4net:
            self.net = C4Net(hid_dim=64, z_dim=64, has_norm=False)
        else:
            self.net = ResNet12Small(avg_pool=True, drop_rate=0.1)

        if "Linux" in platform.platform():
            self.data_root = '/mnt/4T/Data/data/miniImagenet'
            if not os.path.isdir(self.data_root):
                self.data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
        else:
            self.data_root = "F:\\data\\miniImagenet"
        self.data_root = os.path.join(self.data_root, "miniImageNet_png")
        Tools.print(self.data_root)
        Tools.print(self.features_dir)
        pass

    pass


if __name__ == '__main__':
    checkpoint_dir_list = [
        ["../models_abl/miniimagenet/mn/cluster/1_cluster_conv4_300_64_5_1_100_100_png.pkl", True],
        ["../models_abl/miniimagenet/mn/cluster/2_cluster_res12_300_32_5_1_100_100_png.pkl", False],

        ["../models_abl/miniimagenet/mn/css/2_css_conv4_300_64_5_1_100_100_png.pkl", True],
        ["../models_abl/miniimagenet/mn/css/2_css_res12_300_32_5_1_100_100_png.pkl", False],

        ["../models_abl/miniimagenet/mn/label/2_400_64_10_1_200_100_png.pkl", True],
        ["../models_abl/miniimagenet/mn/label/2_100_32_BasicBlock1_0.01_norm2_png_pn_5_1.pkl", False],

        ["../models_abl/miniimagenet/mn/random/2_random_conv4_300_64_5_1_100_100_png.pkl", True],
        ["../models_abl/miniimagenet/mn/random/1_random_res12_300_32_5_1_100_100_png.pkl", False],

        ["../models_abl/miniimagenet/mn/ufsl/3_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn.pkl", True],
        ["../models_abl/miniimagenet/mn/ufsl/1_R12S_1500_32_5_1_300_200_512_1_1.0_1.0_head_png_mn.pkl", False],
    ]
    for checkpoint_dir, is_c4net in checkpoint_dir_list:
        features_dir = checkpoint_dir.replace("models_abl", "vis").replace(".pkl", "")
        config = Config(checkpoint_dir=checkpoint_dir, features_dir=features_dir, is_c4net=is_c4net)
        runner = Runner(config=config)
        runner.load_model()

        runner.features(split="train")
        runner.features(split="val")
        runner.features(split="test")
        pass

    pass
