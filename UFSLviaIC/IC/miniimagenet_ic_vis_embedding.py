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


##############################################################################################################


class MyTSNE(object):

    def __init__(self):
        self.features = Tools.read_from_pkl(Config.feature_dir)

        self.data,  self.label = self.deal_feature()
        pass

    def deal_feature(self):
        data_list, label_list = [], []
        for feature in self.features:
            index, label, cluster, logits, l2norm = feature
            data_list.append(l2norm if Config.is_l2norm else logits)
            label_list.append(cluster if Config.is_cluster else label)
            pass
        return data_list, label_list

    def main(self):
        tsne = TSNE(n_components=2)

        Tools.print("begin to fit_transform {}".format(Config.result_png))
        result = tsne.fit_transform(self.data)

        Tools.print("begin to embedding {}".format(Config.result_png))
        fig = self.plot_embedding(result, self.label, Config.title, Config.s)

        Tools.print("begin to save {}".format(Config.result_png))
        plt.savefig(Config.result_png)
        # plt.show(fig)
        pass

    @staticmethod
    def plot_embedding(data, label, title, s=None):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        # colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        color = list(mcolors.BASE_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())
        fig = plt.figure()
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], s if s is not None else str(label[i]),
                     color=color[label[i]], fontdict={'weight': 'bold', 'size': 6})
            pass

        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig

    pass


##############################################################################################################


class Config(object):
    title = "train"
    s = None

    is_l2norm = True
    is_cluster = False

    split = "train"
    vis_dir = "../vis/miniImagenet/ic_res_xx/3_resnet_34_64_512_1_2100_500_200"
    feature_dir = os.path.join(vis_dir, "{}.pkl".format(split))
    result_png = os.path.join(vis_dir, "{}_{}_{}_{}.png".format(
        split, "l2norm" if is_l2norm else "logits", "cluster" if is_cluster else "label", title))
    pass


if __name__ == '__main__':
    MyTSNE().main()
    pass
