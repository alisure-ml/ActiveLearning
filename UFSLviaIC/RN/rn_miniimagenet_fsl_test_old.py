import os
import math
import torch
import random
import platform
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from rn_miniimagenet_fsl_test_tool import TestTool
from torch.utils.data import DataLoader, Dataset


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

        self.test_tool = TestTool(self.compare_fsl, data_root=Config.data_root,
                                  num_way=Config.num_way,  num_shot=Config.num_shot,
                                  episode_size=Config.episode_size, test_episode=Config.test_episode)
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
                                    batch_features_ext), 2).view(-1, feature_dim * 2, feature_width, feature_height)

        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, Config.num_way * Config.num_shot)
        return relations

    pass


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_way = 5
    num_shot = 1
    episode_size = 15
    batch_size = 1

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

    runner.test_tool.val(episode=0, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=0, is_print=True)
    pass
