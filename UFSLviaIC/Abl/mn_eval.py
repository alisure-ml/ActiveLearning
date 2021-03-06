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
from torchvision.models import resnet18, resnet34
sys.path.append("../Common")
from UFSLTool import MyTransforms, MyDataset, TrainDataset, C4Net, Normalize, ProduceClass
from UFSLTool import RunnerTool, ResNet12Small, ICResNet, FSLTestTool, ICTestTool


##############################################################################################################


class Runner(object):

    def __init__(self, config):
        self.config = config

        # model
        self.norm = Normalize(2)
        self.matching_net = RunnerTool.to_cuda(self.config.matching_net)

        # check
        if self.config.is_check:
            return

        # Eval
        self.test_tool_fsl = FSLTestTool(self.matching_test, data_root=self.config.data_root,
                                      num_way=self.config.num_way, num_shot=self.config.num_shot,
                                      episode_size=self.config.episode_size, test_episode=self.config.test_episode,
                                      transform=self.config.transform_test, txt_path=self.config.log_file)
        pass

    def load_model(self):
        mn_dir = self.config.mn_checkpoint
        if os.path.exists(mn_dir):
            self.matching_net.load_state_dict(torch.load(mn_dir))
            Tools.print("load matching net success from {}".format(mn_dir), txt_path=self.config.log_file)
        pass

    def matching_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)  # 5x64*5*5
        batch_z = self.matching_net(batches)  # 75x64*5*5
        z_support = sample_z.view(num_way * num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, num_way * num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, num_way * num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, num_way, num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    pass


##############################################################################################################


class Config(object):

    def __init__(self, gpu_id=1, name=None, is_conv_4=True, mn_checkpoint=None,
                 dataset_name=MyDataset.dataset_name_miniimagenet, is_check=False):
        self.gpu_id = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.name = name
        self.is_conv_4 = is_conv_4
        self.dataset_name = dataset_name
        self.num_way = 5
        self.num_shot = 1
        self.num_workers = 8
        self.episode_size = 15
        self.test_episode = 600
        self.mn_checkpoint = mn_checkpoint

        ###############################################################################################
        if self.is_conv_4:
            self.matching_net, self.batch_size = C4Net(hid_dim=64, z_dim=64), 64
        else:
            self.matching_net, self.batch_size = ResNet12Small(avg_pool=True, drop_rate=0.1), 32
        ###############################################################################################

        self.is_check = is_check
        if self.is_check:
            self.log_file = None
            return

        self.log_file = Tools.new_dir(os.path.join("../models_abl/{}/mn/result".format(self.dataset_name),
                                                   "{}_{}.txt".format(self.name, Tools.get_format_time())))

        ###############################################################################################
        self.is_png = True
        self.data_root = MyDataset.get_data_root(dataset_name=self.dataset_name, is_png=self.is_png)
        _, _, self.transform_test = MyTransforms.get_transform(
            dataset_name=self.dataset_name, has_ic=True, is_fsl_simple=True, is_css=False)
        ###############################################################################################
        pass

    pass


##############################################################################################################


def final_eval(gpu_id, name, mn_checkpoint, dataset_name, is_conv_4, test_episode=1000):
    config = Config(gpu_id, dataset_name=dataset_name, is_conv_4=is_conv_4, name=name, mn_checkpoint=mn_checkpoint)
    runner = Runner(config=config)

    runner.load_model()
    runner.matching_net.eval()

    ways, shots = MyDataset.get_ways_shots(dataset_name=dataset_name)
    for index, way in enumerate(ways):
        Tools.print("{}/{} way={}".format(index, len(ways), way))
        m, pm = runner.test_tool_fsl.eval(num_way=way, num_shot=1, episode_size=15, test_episode=test_episode)
        Tools.print("way={},shot=1,acc={},con={}".format(way, m, pm), txt_path=config.log_file)
    for index, shot in enumerate(shots):
        Tools.print("{}/{} shot={}".format(index, len(shots), shot))
        m, pm = runner.test_tool_fsl.eval(num_way=5, num_shot=shot, episode_size=15, test_episode=test_episode)
        Tools.print("way=5,shot={},acc={},con={}".format(shot, m, pm), txt_path=config.log_file)
    pass


def miniimagenet_final_eval(gpu_id=0):
    dataset_name = MyDataset.dataset_name_miniimagenet
    checkpoint_path = "../models_abl/{}/mn".format(dataset_name)

    param_list = [
        {"name": "cluster_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "cluster", "1_cluster_conv4_300_64_5_1_100_100_png.pkl")},
        {"name": "css_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "css", "2_css_conv4_300_64_5_1_100_100_png.pkl")},
        {"name": "random_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "random", "2_random_conv4_300_64_5_1_100_100_png.pkl")},
        {"name": "label_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "label", "2_400_64_10_1_200_100_png.pkl")},
        {"name": "ufsl_res18_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "ufsl", "1_2100_64_5_1_500_200_512_1_1.0_1.0_mn.pkl")},
        {"name": "ufsl_res34head_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "ufsl", "3_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn.pkl")},

        {"name": "cluster_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "cluster", "2_cluster_res12_300_32_5_1_100_100_png.pkl")},
        {"name": "css_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "css", "2_css_res12_300_32_5_1_100_100_png.pkl")},
        {"name": "random_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "random", "1_random_res12_300_32_5_1_100_100_png.pkl")},
        {"name": "label_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "label", "2_100_32_BasicBlock1_0.01_norm2_png_pn_5_1.pkl")},
        {"name": "ufsl_res34head_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "ufsl", "1_R12S_1500_32_5_1_300_200_512_1_1.0_1.0_head_png_mn.pkl")},
        {"name": "ufsl_res34head_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "ufsl", "2_R12S_1500_32_5_1_500_200_512_1_1.0_1.0_head_png_mn.pkl")},
    ]

    for index, param in enumerate(param_list):
        Tools.print("Check: {} / {}".format(index, len(param_list)))
        Runner(config=Config(gpu_id, dataset_name=dataset_name, is_conv_4=param["is_conv_4"],
                             name=param["name"], mn_checkpoint=param["mn"], is_check=True)).load_model()
        pass

    for index, param in enumerate(param_list):
        Tools.print("{} / {}".format(index, len(param_list)))
        final_eval(gpu_id, name=param["name"], mn_checkpoint=param["mn"],
                   dataset_name=dataset_name, is_conv_4=param["is_conv_4"])
        pass

    pass


##############################################################################################################


if __name__ == '__main__':
    miniimagenet_final_eval()
    pass
