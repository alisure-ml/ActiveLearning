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
from UFSLTool import MyTransforms, MyDataset, C4Net, Normalize, RunnerTool
from UFSLTool import ResNet12Small, FSLEvalTool, EvalFeatureDataset


##############################################################################################################


class Runner(object):

    def __init__(self, config):
        self.config = config

        # model
        self.norm = Normalize(2)
        self.matching_net = RunnerTool.to_cuda(self.config.matching_net)
        pass

    def load_model(self):
        mn_dir = self.config.mn_checkpoint
        if os.path.exists(mn_dir):
            checkpoint = torch.load(mn_dir)
            if "module." in list(checkpoint.keys())[0]:
                checkpoint = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}
            self.matching_net.load_state_dict(checkpoint)
            Tools.print("load matching net success from {}".format(mn_dir), txt_path=self.config.log_file)
        pass

    def matching_test(self, sample_z, batch_z, num_way, num_shot):
        batch_num, _, _, _ = batch_z.shape

        # sample_z = self.matching_net(samples)  # 5x64*5*5
        # batch_z = self.matching_net(batches)  # 75x64*5*5
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

    def get_test_tool(self, image_features):
        test_tool_fsl = FSLEvalTool(model_fn=self.matching_test, data_root=self.config.data_root,
                                    num_way=self.config.num_way, num_shot=self.config.num_shot,
                                    episode_size=self.config.episode_size, test_episode=self.config.test_episode,
                                    image_features=image_features, txt_path=self.config.log_file)
        return test_tool_fsl

    def get_features(self):
        Tools.print("get_features")

        output_feature = {}
        with torch.no_grad():
            self.matching_net.eval()

            _, _, transform_test = MyTransforms.get_transform(
                dataset_name=self.config.dataset_name, has_ic=True, is_fsl_simple=True, is_css=False)
            data_test = MyDataset.get_data_split(self.config.data_root, split=self.config.split)
            loader = DataLoader(EvalFeatureDataset(data_test, transform_test),
                                self.config.batch_size, False, num_workers=self.config.num_workers)
            for image, image_name in tqdm(loader):
                output = self.matching_net(image.cuda()).data.cpu().numpy()
                for output_one, image_name_one in zip(output, image_name):
                    output_feature[image_name_one] = output_one
                pass
            pass
        return output_feature

    pass


##############################################################################################################


class Config(object):

    def __init__(self, gpu_id=1, name=None, is_conv_4=True, mn_checkpoint=None,
                 dataset_name=MyDataset.dataset_name_miniimagenet, result_dir="result",
                 split=MyDataset.dataset_split_test, is_check=False):
        self.gpu_id = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.name = name
        self.split = split
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

        self.log_file = Tools.new_dir(os.path.join("../models_abl/{}/mn/{}".format(self.dataset_name, result_dir),
                                                   "{}_{}_{}.txt".format(split, self.name, Tools.get_format_time())))

        ###############################################################################################
        self.is_png = True
        self.data_root = MyDataset.get_data_root(dataset_name=self.dataset_name, is_png=self.is_png)
        _, _, self.transform_test = MyTransforms.get_transform(
            dataset_name=self.dataset_name, has_ic=True, is_fsl_simple=True, is_css=False)
        ###############################################################################################
        pass

    pass


##############################################################################################################


def final_eval(gpu_id, name, mn_checkpoint, dataset_name, is_conv_4,
               test_episode=1000, result_dir="result", split=MyDataset.dataset_split_test, ways_and_shots=None):
    config = Config(gpu_id, dataset_name=dataset_name, is_conv_4=is_conv_4,
                    name=name, mn_checkpoint=mn_checkpoint, result_dir=result_dir, split=split)
    runner = Runner(config=config)

    runner.load_model()
    runner.matching_net.eval()
    image_features = runner.get_features()
    test_tool_fsl = runner.get_test_tool(image_features=image_features)

    if ways_and_shots is None:
        ways, shots = MyDataset.get_ways_shots(dataset_name=dataset_name, split=split)
        for index, way in enumerate(ways):
            Tools.print("{}/{} way={}".format(index, len(ways), way))
            m, pm = test_tool_fsl.eval(num_way=way, num_shot=1, episode_size=15, test_episode=test_episode, split=split)
            Tools.print("way={},shot=1,acc={},con={}".format(way, m, pm), txt_path=config.log_file)
        for index, shot in enumerate(shots):
            Tools.print("{}/{} shot={}".format(index, len(shots), shot))
            m, pm = test_tool_fsl.eval(num_way=5, num_shot=shot, episode_size=15, test_episode=test_episode, split=split)
            Tools.print("way=5,shot={},acc={},con={}".format(shot, m, pm), txt_path=config.log_file)
    else:
        for index, (way, shot) in enumerate(ways_and_shots):
            Tools.print("{}/{} way={} shot={}".format(index, len(ways_and_shots), way, shot))
            m, pm = test_tool_fsl.eval(num_way=way, num_shot=shot, episode_size=15, test_episode=test_episode, split=split)
            Tools.print("way={},shot={},acc={},con={}".format(way, shot, m, pm), txt_path=config.log_file)
            pass

    pass


def miniimagenet_final_eval(gpu_id=0, result_dir="result"):
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
                   dataset_name=dataset_name, is_conv_4=param["is_conv_4"], result_dir=result_dir)
        pass

    pass


def miniimagenet_our_eval(gpu_id=0, result_dir="result_our"):
    dataset_name = MyDataset.dataset_name_miniimagenet
    checkpoint_path = "../models_abl/{}/mn".format(dataset_name)

    ways_and_shots = [[5, 1], [5, 5], [20, 1], [20, 5]]
    param_list = [
        {"name": "cluster_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "cluster", "1_cluster_conv4_300_64_5_1_100_100_png.pkl")},
        {"name": "css_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "css", "2_css_conv4_300_64_5_1_100_100_png.pkl")},
        {"name": "random_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "random", "2_random_conv4_300_64_5_1_100_100_png.pkl")},
        {"name": "label_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "label", "2_400_64_10_1_200_100_png.pkl")},
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
         "mn": os.path.join(checkpoint_path, "ufsl", "2_R12S_1500_32_5_1_500_200_512_1_1.0_1.0_head_png_mn.pkl")},
    ]

    for index, param in enumerate(param_list):
        Tools.print("Check: {} / {}".format(index, len(param_list)))
        Runner(config=Config(gpu_id, dataset_name=dataset_name, is_conv_4=param["is_conv_4"],
                             name=param["name"], mn_checkpoint=param["mn"], is_check=True)).load_model()
        pass

    for index, param in enumerate(param_list):
        Tools.print("{} / {}".format(index, len(param_list)))
        final_eval(gpu_id, name=param["name"], mn_checkpoint=param["mn"], dataset_name=dataset_name,
                   is_conv_4=param["is_conv_4"], result_dir=result_dir,
                   split=MyDataset.dataset_split_test, ways_and_shots=ways_and_shots)
        pass

    pass


def tieredimagenet_final_eval(gpu_id=0, result_dir="result"):
    dataset_name = MyDataset.dataset_name_tieredimagenet
    checkpoint_path = "../models_abl/{}/mn".format(dataset_name)

    param_list = [
        {"name": "cluster_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "cluster", "2_cluster_conv4_150_64_5_1_80_120_png.pkl")},
        {"name": "css_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "css", "1_css_conv4_150_64_5_1_80_120_png.pkl")},
        {"name": "random_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "random", "2_random_conv4_150_64_5_1_80_120_png.pkl")},
        {"name": "label_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "label", "0123_100_256_5_1_conv4.pkl")},
        {"name": "ufsl_res18_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "ufsl", "0123_res18_1200_1024_2048_conv4_200_5_1_384_mn.pkl")},
        # {"name": "ufsl_res34head_conv4", "is_conv_4": True,
        #  "mn": os.path.join(checkpoint_path, "ufsl", "3_2100_64_5_1_500_200_512_1_1.0_1.0_head_png_mn.pkl")},

        {"name": "cluster_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "cluster", "2_cluster_res12_30_32_5_1_16_24_png.pkl")},
        {"name": "css_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "css", "1_css_res12_50_32_5_1_30_40_png.pkl")},
        {"name": "random_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "random", "1_random_res12_30_32_5_1_16_8_png.pkl")},
        {"name": "label_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "label", "0123_100_64_5_1_res12.pkl")},
        {"name": "ufsl_res18_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "ufsl", "0123_res18_1200_1024_2048_resnet12_100_5_1_128_mn.pkl")},
        # {"name": "ufsl_res34head_res12", "is_conv_4": False,
        #  "mn": os.path.join(checkpoint_path, "ufsl", "2_R12S_1500_32_5_1_500_200_512_1_1.0_1.0_head_png_mn.pkl")},
    ]

    for index, param in enumerate(param_list):
        Tools.print("Check: {} / {}".format(index, len(param_list)))
        Runner(config=Config(gpu_id, dataset_name=dataset_name, is_conv_4=param["is_conv_4"],
                             name=param["name"], mn_checkpoint=param["mn"], is_check=True)).load_model()
        pass

    for index, param in enumerate(param_list):
        Tools.print("{} / {}".format(index, len(param_list)))
        final_eval(gpu_id, name=param["name"], mn_checkpoint=param["mn"],
                   dataset_name=dataset_name, is_conv_4=param["is_conv_4"], result_dir=result_dir)
        pass

    pass


def tieredimagenet_our_eval(gpu_id=0, result_dir="result_our"):
    dataset_name = MyDataset.dataset_name_tieredimagenet
    checkpoint_path = "../models_abl/{}/mn".format(dataset_name)

    ways_and_shots = [[5, 1], [5, 5], [20, 1], [20, 5]]
    param_list = [
        {"name": "cluster_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "cluster", "2_cluster_conv4_150_64_5_1_80_120_png.pkl")},
        {"name": "css_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "css", "1_css_conv4_150_64_5_1_80_120_png.pkl")},
        {"name": "random_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "random", "2_random_conv4_150_64_5_1_80_120_png.pkl")},
        {"name": "label_conv4", "is_conv_4": True,
         "mn": os.path.join(checkpoint_path, "label", "0123_100_256_5_1_conv4.pkl")},
        # {"name": "ufsl_res18_conv4", "is_conv_4": True,
        #  "mn": os.path.join(checkpoint_path, "ufsl", "0123_res18_1200_1024_2048_conv4_200_5_1_384_mn.pkl")},

        {"name": "cluster_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "cluster", "2_cluster_res12_30_32_5_1_16_24_png.pkl")},
        {"name": "css_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "css", "1_css_res12_50_32_5_1_30_40_png.pkl")},
        {"name": "random_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "random", "1_random_res12_30_32_5_1_16_8_png.pkl")},
        {"name": "label_res12", "is_conv_4": False,
         "mn": os.path.join(checkpoint_path, "label", "0123_100_64_5_1_res12.pkl")},
        # {"name": "ufsl_res18_res12", "is_conv_4": False,
        #  "mn": os.path.join(checkpoint_path, "ufsl", "0123_res18_1200_1024_2048_resnet12_100_5_1_128_mn.pkl")},
    ]

    for index, param in enumerate(param_list):
        Tools.print("Check: {} / {}".format(index, len(param_list)))
        Runner(config=Config(gpu_id, dataset_name=dataset_name, is_conv_4=param["is_conv_4"],
                             name=param["name"], mn_checkpoint=param["mn"], is_check=True)).load_model()
        pass

    for index, param in enumerate(param_list):
        Tools.print("{} / {}".format(index, len(param_list)))
        final_eval(gpu_id, name=param["name"], mn_checkpoint=param["mn"], dataset_name=dataset_name,
                   is_conv_4=param["is_conv_4"], result_dir=result_dir,
                   split=MyDataset.dataset_split_test, ways_and_shots=ways_and_shots)
        pass

    pass


##############################################################################################################


if __name__ == '__main__':
    miniimagenet_our_eval(result_dir="result_table")
    # tieredimagenet_final_eval()
    tieredimagenet_our_eval(result_dir="result_table")
    pass
