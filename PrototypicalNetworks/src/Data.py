import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from functools import partial
from torchvision.transforms import ToTensor
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose as TransformCompose


class TransformRotateImage(object):

    def __init__(self, rot):
        self.rot = rot
        pass

    def __call__(self, data):
        return data.rotate(self.rot)

    pass


class TransformScaleImage(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width
        pass

    def __call__(self, data):
        return data.resize((self.height, self.width))

    pass


class TransformConvertTensor(object):

    def __call__(self, data):
        return 1.0 - torch.from_numpy(np.array(data, np.float32, copy=False)
                                      ).transpose(0, 1).contiguous().view(1, data.size[0], data.size[1])

    pass


class TransformLoadImage(object):

    def __call__(self, file_name):
        return Image.open(file_name)

    pass


class TransformExtractEpisode(object):

    def __init__(self, n_support, n_query):
        self.n_support = n_support
        self.n_query = n_query
        pass

    def __call__(self, data_list):
        n_examples = data_list.size(0)
        if self.n_query == -1:
            self.n_query = n_examples - self.n_support
            pass

        example_indexes = torch.randperm(n_examples)[:(self.n_support + self.n_query)]
        support_indexes = example_indexes[:self.n_support]
        query_indexes = example_indexes[self.n_support:]
        return {'xs': data_list[support_indexes], 'xq': data_list[query_indexes]}

    pass


class SequentialBatchSampler(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        pass

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])
        pass

    pass


class EpisodicBatchSampler(object):

    def __init__(self, n_classes, n_way, n_episodes):
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_episodes = n_episodes
        pass

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
        pass

    pass


class Omniglot(object):

    def __init__(self, data_dir="D:\\Data\\omniglot",
                 split_dir="D:\\Pycharm\\File\\ActiveLearning\\PrototypicalNetworks\\data\\omniglot\\splits"):
        self.data_dir = data_dir
        self.split_dir = split_dir

        self.cache = {}
        pass

    @staticmethod
    def read_class_names(split_dir, split):
        class_names = []
        with open(os.path.join(split_dir, "{:s}.txt".format(split)), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))
                pass
            pass
        return class_names

    # 从类名到数据
    def load_class_images(self, class_name):
        if class_name not in self.cache:
            alphabet, character, rot = class_name.split('/')
            image_dir = os.path.join(self.data_dir, 'data', alphabet, character)
            class_images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            assert len(class_images) > 0

            image_ds = TransformDataset(ListDataset(class_images),
                                        TransformCompose([TransformLoadImage(),
                                                          TransformRotateImage(rot=float(rot[3:])),
                                                          TransformScaleImage(height=28, width=28),
                                                          TransformConvertTensor()]))
            for sample in torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False):
                self.cache[class_name] = sample
                break  # only need one sample because batch size equal to dataset length
            pass
        return self.cache[class_name]

    def load(self, config, splits):
        split_dir = os.path.join(self.split_dir, config.data.split)

        ret = {}
        for split in splits:
            which = split in ['val', 'test']
            n_way = config.data.test_way if which and config.data.test_way != 0 else config.data.way
            n_support = config.data.test_shot if which and config.data.test_shot != 0 else config.data.shot
            n_query = config.data.test_query if which and config.data.test_query != 0 else config.data.query
            n_episodes = config.data.test_episodes if which else config.data.train_episodes

            class_names = self.read_class_names(split_dir=split_dir, split=split)
            dataset = TransformDataset(ListDataset(class_names), TransformCompose([
                self.load_class_images, TransformExtractEpisode(n_support=n_support, n_query=n_query)]))
            if config.data.sequential:
                sampler = SequentialBatchSampler(len(dataset))
            else:
                sampler = EpisodicBatchSampler(len(dataset), n_way, n_episodes)

            ret[split] = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)
            pass
        return ret

    pass


class MyData(object):

    @staticmethod
    def load(config, splits):
        if config.data.dataset == 'omniglot':
            ds = Omniglot().load(config, splits)
        else:
            raise ValueError("Unknown dataset: {:s}".format(config.data.dataset))
        return ds

    pass
