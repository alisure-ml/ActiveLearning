import os
import math
import torch
import random
import platform
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from alisuretool.Tools import Tools
from pn_miniimagenet_fsl_test_tool import TestTool
from pn_miniimagenet_tool import ProtoNet, RunnerTool


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.proto_net = RunnerTool.to_cuda(Config.proto_net)
        pass

    def load_model(self):
        if os.path.exists(Config.pn_dir):
            self.proto_net.load_state_dict(torch.load(Config.pn_dir))
            Tools.print("load proto network success from {}".format(Config.pn_dir))
        pass

    def compare_fsl_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        sample_z = self.proto_net(samples)  # 5x64*5*5
        batch_z = self.proto_net(batches)  # 75x64*5*5
        sample_z = sample_z.view(num_way, num_shot, -1)
        batch_z = batch_z.view(batch_num, -1)
        _, z_dim = batch_z.shape

        z_proto = sample_z.mean(1)
        z_proto_expand = z_proto.unsqueeze(0).expand(batch_num, num_way, z_dim)
        z_query_expand = batch_z.unsqueeze(1).expand(batch_num, num_way, z_dim)

        dists = torch.pow(z_query_expand - z_proto_expand, 2).sum(2)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def eval_one(self, num_way=5, num_shot=1):
        Tools.print("way:{} shot:{}".format(num_way, num_shot))

        self.proto_net.eval()
        test_tool_fsl = TestTool(self.compare_fsl_test, Config.data_root, num_way=num_way, num_shot=num_shot,
                                 episode_size=Config.episode_size, test_episode=Config.test_episode)
        test_tool_fsl.val(is_print=True)
        test_tool_fsl.test(test_avg_num=5, is_print=True)
        pass

    def eval(self):
        self.eval_one(num_way=5, num_shot=1)
        self.eval_one(num_way=5, num_shot=5)
        self.eval_one(num_way=5, num_shot=10)
        self.eval_one(num_way=5, num_shot=20)
        self.eval_one(num_way=10, num_shot=1)
        self.eval_one(num_way=10, num_shot=5)
        self.eval_one(num_way=10, num_shot=10)
        self.eval_one(num_way=10, num_shot=20)
        pass

    pass


##############################################################################################################


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8
    episode_size = 15
    test_episode = 600
    test_avg_num = 5

    hid_dim = 64
    z_dim = 64

    # has_norm = True
    has_norm = False
    proto_net = ProtoNet(hid_dim=hid_dim, z_dim=z_dim, has_norm=has_norm)

    model_path = "../models_pn/two_ic_ufsl_2net_res_sgd_acc_duli"
    # model_pn_name = "2_2100_64_5_1_64_64_500_200_512_1_1.0_1.0_norm_pn_5way_1shot.pkl"
    model_pn_name = "1_2100_64_5_1_64_64_500_200_512_1_1.0_1.0__pn_5way_1shot.pkl"
    pn_dir = Tools.new_dir(os.path.join(model_path, model_pn_name))

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
