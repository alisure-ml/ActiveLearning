import os
import cv2
import json
import random
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from alisuretool.Tools import Tools


def load_split(data_root, split):
    Tools.print("load {} data".format(split))

    images_npz_file = os.path.join(data_root, "{}_images_png.npz".format(split))
    if os.path.exists(images_npz_file):
        Tools.print("exist {} and load".format(images_npz_file))
        _images = np.load(images_npz_file)['images']
    else:
        images_pkl_file = os.path.join(data_root, "{}_images_png.pkl".format(split))
        images_pkl = Tools.read_from_pkl(images_pkl_file)

        _images = np.zeros([len(images_pkl), 84, 84, 3], dtype=np.uint8)
        for ii, item in tqdm(enumerate(images_pkl), desc='decompress'):
            _images[ii] = cv2.imdecode(item, 1)
            pass
        np.savez(images_npz_file, images=_images)
        pass

    labels_npz_file = os.path.join(data_root, "{}_labels.npz".format(split))
    if os.path.exists(labels_npz_file):
        Tools.print("exist {} and load".format(labels_npz_file))
        _labels = np.load(labels_npz_file)['labels']
    else:
        labels_pkl_file = os.path.join(data_root, "{}_labels.pkl".format(split))
        labels_pkl = Tools.read_from_pkl(labels_pkl_file)
        _labels = labels_pkl["label_specific"]
        _labels = _labels - np.min(_labels)
        np.savez(labels_npz_file, labels=_labels)
        pass
    return _images, _labels


def save_image(data_root, split, result_path):
    Tools.print("load {} data".format(split))
    images_pkl_file = os.path.join(data_root, "{}_images_png.pkl".format(split))
    labels_pkl_file = os.path.join(data_root, "{}_labels.pkl".format(split))
    images_pkl = Tools.read_from_pkl(images_pkl_file)
    labels_pkl = Tools.read_from_pkl(labels_pkl_file)

    for ii, item in tqdm(enumerate(images_pkl), desc='decompress'):
        image = cv2.imdecode(item, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_file = Tools.new_dir(os.path.join(result_path, split,
                                                 str(labels_pkl["label_specific"][ii]), "{}.png".format(ii)))
        Image.fromarray(image).save(result_file)
        pass

    pass


if __name__ == '__main__':
    # data_dir = "/media/ubuntu/4T/ALISURE/Data/UFSL/tiered-imagenet"
    data_dir = "/mnt/4T/Data/data/UFSL/tiered-imagenet"
    # train_images, train_labels = load_split(data_dir, split="train")
    # test_images, test_labels = load_split(data_dir, split="test")
    # val_images, val_labels = load_split(data_dir, split="val")
    save_image(data_dir, split="val", result_path=data_dir)
    save_image(data_dir, split="train", result_path=data_dir)
    save_image(data_dir, split="test", result_path=data_dir)
    pass
