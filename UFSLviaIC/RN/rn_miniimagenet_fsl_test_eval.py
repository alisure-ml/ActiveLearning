import os
import math
import torch
import random
import platform
import numpy as np
import torch.nn as nn
from PIL import Image
from alisuretool.Tools import Tools
from rn_miniimagenet_fsl_test_tool import TestTool
from rn_miniimagenet_tool import CNNEncoder, RelationNetwork, RunnerTool


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.feature_encoder = RunnerTool.to_cuda(Config.feature_encoder)
        self.relation_network = RunnerTool.to_cuda(Config.relation_network)
        pass

    def load_model(self):
        if os.path.exists(Config.fe_dir):
            self.feature_encoder.load_state_dict(torch.load(Config.fe_dir))
            Tools.print("load feature encoder success from {}".format(Config.fe_dir))

        if os.path.exists(Config.rn_dir):
            self.relation_network.load_state_dict(torch.load(Config.rn_dir))
            Tools.print("load relation network success from {}".format(Config.rn_dir))
        pass

    def compare_fsl_test(self, samples, batches):
        # calculate features
        sample_features = self.feature_encoder(samples)  # 5x64*19*19
        num_shot_num_way = sample_features.shape[0]
        batch_features = self.feature_encoder(batches)  # 75x64*19*19
        batch_size, feature_dim, feature_width, feature_height = batch_features.shape

        # calculate
        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(1).repeat(1, num_shot_num_way, 1, 1, 1)

        # calculate relations
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
        relation_pairs = relation_pairs.view(-1, feature_dim * 2, feature_width, feature_height)

        relations = self.relation_network(relation_pairs)
        relations = relations.view(-1, num_shot_num_way)
        return relations

    def eval_one(self, num_way=5, num_shot=1):
        Tools.print("way:{} shot:{}".format(num_way, num_shot))

        test_tool_fsl = TestTool(self.compare_fsl_test, Config.data_root, num_way=num_way, num_shot=num_shot,
                                 episode_size=Config.episode_size, test_episode=Config.test_episode)
        test_tool_fsl.val(is_print=True)
        test_tool_fsl.test(test_avg_num=5, is_print=True)
        pass

    def eval(self):
        self.eval_one(num_way=5, num_shot=1)
        self.eval_one(num_way=5, num_shot=5)
        self.eval_one(num_way=5, num_shot=10)
        self.eval_one(num_way=10, num_shot=1)
        self.eval_one(num_way=10, num_shot=5)
        pass

    pass


##############################################################################################################


"""

"""


class Config(object):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_workers = 8
    batch_size = 64
    episode_size = 15
    test_episode = 600
    test_avg_num = 5

    feature_encoder, relation_network = CNNEncoder(), RelationNetwork()

    model_path = "../models/two_ic_ufsl_2net_res_sgd_acc"
    model_fe_name = "0_2100_64_5_1_500_200_512_1_1.0_1.0_fe_5way_1shot.pkl"
    model_rn_name = "0_2100_64_5_1_500_200_512_1_1.0_1.0_rn_5way_1shot.pkl"
    fe_dir = Tools.new_dir(os.path.join(model_path, model_fe_name))
    rn_dir = Tools.new_dir(os.path.join(model_path, model_rn_name))

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pass


if __name__ == '__main__':
    runner = Runner()

    runner.load_model()
    runner.eval()
    pass
