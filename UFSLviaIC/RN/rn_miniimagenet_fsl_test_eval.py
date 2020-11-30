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

        # self.feature_encoder.eval()
        # self.relation_network.eval()

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


"""
2020-11-26 13:05:06 way:5 shot:5
2020-11-26 13:08:36 Train 0 Accuracy: 0.5961333333333333
2020-11-26 13:08:36 Val   0 Accuracy: 0.5547333333333334
2020-11-26 13:08:36 Test1 0 Accuracy: 0.5676666666666667
2020-11-26 13:08:36 Test2 0 Accuracy: 0.5621111111111111
2020-11-26 13:15:57 episode=0, Mean Test accuracy=0.5683555555555555
2020-11-26 13:15:57 way:5 shot:1
2020-11-26 13:17:53 Train 0 Accuracy: 0.4935555555555555
2020-11-26 13:17:53 Val   0 Accuracy: 0.4393333333333333
2020-11-26 13:17:53 Test1 0 Accuracy: 0.458
2020-11-26 13:17:53 Test2 0 Accuracy: 0.46271111111111113
2020-11-26 13:22:43 episode=0, Mean Test accuracy=0.46001777777777775
"""


"""
2020-11-26 13:11:06 way:5 shot:1
2020-11-26 13:12:50 Train 0 Accuracy: 0.49344444444444446
2020-11-26 13:12:50 Val   0 Accuracy: 0.45133333333333336
2020-11-26 13:17:01 episode=0, Mean Test accuracy=0.46661777777777774
2020-11-26 13:17:01 way:5 shot:5
2020-11-26 13:19:45 Train 0 Accuracy: 0.6024666666666666
2020-11-26 13:19:45 Val   0 Accuracy: 0.5617333333333333
2020-11-26 13:25:14 episode=0, Mean Test accuracy=0.5753777777777778
2020-11-26 13:25:14 way:5 shot:10
2020-11-26 13:29:06 Train 0 Accuracy: 0.6469333333333332
2020-11-26 13:29:06 Val   0 Accuracy: 0.6004
2020-11-26 13:36:32 episode=0, Mean Test accuracy=0.6134666666666666
2020-11-26 13:36:32 way:10 shot:1
2020-11-26 13:39:49 Train 0 Accuracy: 0.3608888888888888
2020-11-26 13:39:49 Val   0 Accuracy: 0.30927777777777776
2020-11-26 13:47:55 episode=0, Mean Test accuracy=0.3181622222222222
2020-11-26 13:47:55 way:10 shot:5
2020-11-26 13:54:04 Train 0 Accuracy: 0.47339999999999993
2020-11-26 13:54:04 Val   0 Accuracy: 0.4098
2020-11-26 14:06:47 episode=0, Mean Test accuracy=0.4181444444444445
"""


"""
../models/two_ic_ufsl_2net_res_sgd_acc_duli/2_2100_64_5_1_500_200_512_1_1.0_1.0_fe_5way_1shot.pkl
../models/two_ic_ufsl_2net_res_sgd_acc_duli/2_2100_64_5_1_500_200_512_1_1.0_1.0_rn_5way_1shot.pkl
2020-11-26 19:47:28 way:5 shot:1
2020-11-26 19:49:37 Train 0 Accuracy: [0.50055556 0.50055556]
2020-11-26 19:49:37 Val   0 Accuracy: [0.443 0.443]
2020-11-26 19:55:17 episode=0, Mean Test accuracy=[0.46670667 0.46670667]
2020-11-26 19:55:17 way:5 shot:5
2020-11-26 20:00:24 Train 0 Accuracy: [0.65426667 0.61126667]
2020-11-26 20:00:24 Val   0 Accuracy: [0.61186667 0.564     ]
2020-11-26 20:12:32 episode=0, Mean Test accuracy=[0.62052    0.57646667]
2020-11-26 20:12:32 way:5 shot:10
2020-11-26 20:21:07 Train 0 Accuracy: [0.699      0.64433333]
2020-11-26 20:21:07 Val   0 Accuracy: [0.65126667 0.5994    ]
2020-11-26 20:40:21 episode=0, Mean Test accuracy=[0.66968444 0.61347556]
2020-11-26 20:40:21 way:5 shot:20
2020-11-26 20:55:49 Train 0 Accuracy: [0.7176     0.67293333]
2020-11-26 20:55:49 Val   0 Accuracy: [0.686      0.62966667]
2020-11-26 21:31:10 episode=0, Mean Test accuracy=[0.70228    0.64510667]
2020-11-26 21:31:10 way:10 shot:1
2020-11-26 21:36:08 Train 0 Accuracy: [0.35711111 0.35711111]
2020-11-26 21:36:08 Val   0 Accuracy: [0.31277778 0.31277778]
2020-11-26 21:49:22 episode=0, Mean Test accuracy=[0.31871111 0.31871111]
2020-11-26 21:49:22 way:10 shot:5
2020-11-26 22:05:08 Train 0 Accuracy: [0.51143333 0.4663    ]
2020-11-26 22:05:08 Val   0 Accuracy: [0.45333333 0.4126    ]
2020-11-26 22:41:33 episode=0, Mean Test accuracy=[0.46245778 0.41782444]
2020-11-26 22:41:33 way:10 shot:10
2020-11-26 23:09:41 Train 0 Accuracy: [0.55826667 0.5121    ]
2020-11-26 23:09:41 Val   0 Accuracy: [0.49926667 0.4458    ]
2020-11-27 00:16:26 episode=0, Mean Test accuracy=[0.51141111 0.45526222]

"""


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8
    episode_size = 15
    test_episode = 600
    test_avg_num = 5

    feature_encoder, relation_network = CNNEncoder(), RelationNetwork()

    # model_path = "../models/two_ic_ufsl_2net_res_sgd_acc"
    # model_fe_name = "0_2100_64_5_1_500_200_512_1_1.0_1.0_fe_5way_1shot.pkl"
    # model_rn_name = "0_2100_64_5_1_500_200_512_1_1.0_1.0_rn_5way_1shot.pkl"

    model_path = "../models/two_ic_ufsl_2net_res_sgd_acc_duli"
    model_fe_name = "2_2100_64_5_1_500_200_512_1_1.0_1.0_fe_5way_1shot.pkl"
    model_rn_name = "2_2100_64_5_1_500_200_512_1_1.0_1.0_rn_5way_1shot.pkl"

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
