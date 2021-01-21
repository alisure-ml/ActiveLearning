import os
import math
import platform
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
from sklearn.manifold import TSNE
from alisuretool.Tools import Tools
from matplotlib import colors as mcolors

import time
import socket
from datetime import datetime

##############################################################################################################


class MyTSNE(object):

    def __init__(self, config):
        self.config = config
        pass

    def deal_feature(self):

        def get_one_feature(feature_dir, data_list, label_list, label_fixed=None):
            features = Tools.read_from_pkl(feature_dir)
            label_dict, label_dict_count = {}, {}
            for feature in features:
                _, label, _, logits, l2norm = feature

                is_ok = False
                if self.config.class_id is None or label in self.config.class_id:  # 类别
                    if label not in label_dict:
                        label_dict[label] = len(label_dict.keys())
                    if self.config.sample is None:  # 采样
                        is_ok = True
                    else:
                        if label not in label_dict_count:  # 计数
                            label_dict_count[label] = 0
                        if label_dict_count[label] < self.config.sample:
                            label_dict_count[label] += 1
                            is_ok = True
                        pass
                    pass

                if is_ok:
                    data_list.append(l2norm if self.config.is_l2norm else logits)
                    label_list.append(label_dict[label] if label_fixed is None else label_fixed)
                    pass

                pass
            return data_list, label_list

        _data_list, _label_list = [], []
        if self.config.split == "all":
            for split_index, _feature_dir in enumerate(self.config.feature_dir_list):
                _data_list, _label_list = get_one_feature(_feature_dir, _data_list,
                                                          _label_list, label_fixed=split_index)
        else:
            _data_list, _label_list = get_one_feature(self.config.feature_dir, _data_list, _label_list)
            pass
        return _data_list, _label_list

    def main(self):
        tsne = TSNE(n_components=2)

        Tools.print("begin to fit_transform {}".format(self.config.result_png))
        if os.path.exists(self.config.result_pkl):
            Tools.print("exist pkl, and now to load")
            result = Tools.read_from_pkl(self.config.result_pkl)
        else:
            Tools.print("not exist pkl, and now to fit")
            data,  label = self.deal_feature()
            fit = tsne.fit_transform(data)
            result = {"fit": fit, "label": label}
            Tools.write_to_pkl(self.config.result_pkl, result)
            pass

        Tools.print("begin to embedding")
        fig = self.plot_embedding(result["fit"], result["label"])

        Tools.print("begin to save")
        plt.savefig(self.config.result_png)
        # plt.show(fig)
        pass

    @classmethod
    def plot_embedding(cls, data, label):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min) * 0.96  + 0.01

        fig = plt.figure(figsize=(10, 10))
        color_shape = cls.my_color(max(label) + 1)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], color_shape[label[i]][1][0],
                     color=color_shape[label[i]][0], fontdict={'size': color_shape[label[i]][1][1]})
            pass
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.axis('off')
        # plt.xticks([])
        # plt.yticks([])
        # plt.title(title)
        return fig

    @staticmethod
    def my_color(num):
        # color = list(mcolors.BASE_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())

        colors = ["b", "g", "r", "k", "c", "m", "#FFA500","#FFFF00","#800000","#B8860B","#808080","#4B0082"]
        shapes = [ ["●", 8]]
         #FFA500是橙色,FFFF00是黄色,800000深红色,B8860B darkgoldenrod,808080灰色,4B0082 靛蓝色
        #["▴", 10], ["★", 10],["▪", 10],["◆",6],["✚",7],["✚",7],,["◀",6],["▼",6],["▶",6]
        if num > len(colors) * len(shapes):
            count = num // len(shapes) - len(colors) + 1
            colors += list(mcolors.CSS4_COLORS.keys())[: count]
            pass

        result = []
        for shape in shapes:
            result.extend([(color, shape) for color in colors])
        return result

    pass


##############################################################################################################


class Config(object):

    def __init__(self, split="all", class_id=None, is_l2norm=True, result_dir_name=None, png_name=None, sample=None):
        self.title = "vis"

        self.split = split
        self.is_l2norm = is_l2norm
        self.class_id = class_id
        self.sample = sample
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')# or 
        self.vis_dir = "/home/test/Documents/kbzhao/IC/IC_result/3_resnet_34_64_512_1_2100_500_200"
        self.result_pkl = Tools.new_dir(os.path.join(self.vis_dir, "fig_final_new", "{}_{}{}.pkl".format(
            self.split, "l2norm" if self.is_l2norm else "logits",
            "" if png_name is None else "_{}".format(png_name))))
        self.result_png = self.result_pkl[:-4] + f"{current_time}.png"

        split_path = os.path.split(self.result_png)
        self.result_png = Tools.new_dir(os.path.join(split_path[0], result_dir_name, split_path[1]))

        if self.split == "all":
            self.feature_dir_list = [os.path.join(self.vis_dir, "train.pkl"),
                                     os.path.join(self.vis_dir, "test.pkl"),
                                     os.path.join(self.vis_dir, "val.pkl")]
        else:
            self.feature_dir = os.path.join(self.vis_dir, "{}.pkl".format(self.split))
            pass
        pass

    pass


##############################################################################################################


if __name__ == '__main__':
    result_dir_name = "abc4"
    sample = 100
    class_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    split_list = ["val", "test", "train", "all"]

    # for split in split_list:
    #     config = Config(split=split, sample=sample, class_id=class_id, is_l2norm=True,
    #                     result_dir_name=result_dir_name, 
    #                     png_name="1_{}_{}_{}".format(split, sample, len(class_id)))
    #     my_tsne = MyTSNE(config=config)
    #     my_tsne.main()
    #     pass
    
    # for split in split_list:
    #     config = Config(split=split, sample=sample, is_l2norm=True, 
    #                     result_dir_name=result_dir_name, png_name="2_{}_{}".format(split, sample))
    #     my_tsne = MyTSNE(config=config)
    #     my_tsne.main()
    #     pass

    for split in split_list:
        config = Config(split=split, class_id=class_id, is_l2norm=True, 
                        result_dir_name=result_dir_name, png_name="3_{}_{}".format(split, len(class_id)))
        my_tsne = MyTSNE(config=config)
        my_tsne.main()
        pass

    # for split in split_list:
    #     config = Config(split=split, is_l2norm=True, 
    #                     result_dir_name=result_dir_name, png_name="4_{}".format(split))
    #     my_tsne = MyTSNE(config=config)
    #     my_tsne.main()
    #     pass

    pass

