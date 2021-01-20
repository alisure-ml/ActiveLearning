import os
import sys
import time
import faiss
import torch
import platform
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
from alisuretool.Tools import Tools
from scipy.sparse import csr_matrix, find
import torchvision.transforms as transforms
sys.path.append("../Common")
from UFSLTool import MyDataset


"""
pip install faiss-gpu

sudo apt-get install libopenblas-dev
"""


class KMeans(object):

    def __init__(self, k):
        self.k = k
        pass

    def cluster(self, data):
        Tools.print("PCA-reducing, whitening and L2-normalization")
        xb = self.preprocess_features(data)

        Tools.print("cluster the data")
        I = self.run_kmeans(xb, self.k)

        images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            images_lists[I[i]].append(i)
            pass
        return images_lists

    @staticmethod
    def preprocess_features(npdata, pca=256):
        _, ndim = npdata.shape
        npdata = npdata.astype('float32')

        # Apply PCA-whitening with Faiss
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)

        # L2 normalization
        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / row_sums[:, np.newaxis]

        return npdata

    @staticmethod
    def run_kmeans(x, nmb_clusters):
        n_data, d = x.shape

        # faiss implementation of k-means
        clus = faiss.Clustering(d, nmb_clusters)

        # Change faiss seed at each k-means so that the randomly picked
        # initialization centroids do not correspond to the same feature ids
        # from an epoch to another.
        clus.seed = np.random.randint(1234)

        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        # perform the training
        clus.train(x, index)
        _, I = index.search(x, 1)

        return [int(n[0]) for n in I]

    pass


class Config(object):
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    is_png = True

    # dataset_name = "miniimagenet"
    # dataset_name = "tieredimagenet"
    dataset_name = MyDataset.dataset_name_omniglot

    data_root = MyDataset.get_data_root(dataset_name=dataset_name, is_png=is_png)
    Tools.print(data_root)

    features_save_path = Tools.new_dir("{}_feature".format(data_root))
    Tools.print(features_save_path)
    pass


if __name__ == '__main__':
    features_dict = Tools.read_from_pkl(os.path.join(Config.features_save_path, "train_features.pkl"))
    data_list, features = features_dict["info"], np.asarray(features_dict["feature"])

    k_means = KMeans(k=512)
    images_lists = k_means.cluster(features)

    cluster_result = np.zeros(shape=len(data_list), dtype=np.int)
    for cluster_id, cluster_image in enumerate(images_lists):
        cluster_result[cluster_image] = cluster_id
        pass
    Tools.write_to_pkl(os.path.join(Config.features_save_path, "train_cluster.pkl"),
                       {"info": data_list, "cluster": cluster_result})
    pass
