import os
import random
import platform
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from alisuretool.Tools import Tools


if "Linux" in platform.platform():
    data_root = '/mnt/4T/Data/data/UFSL/omniglot_single'
    if not os.path.isdir(data_root):
        data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/omniglot_single'
    if not os.path.isdir(data_root):
        data_root = '/home/ubuntu/Dataset/Partition1/ALISURE/Data/UFSL/omniglot_single'
else:
    data_root = "F:\\data\\omniglot_single"


rot_list = [0, 90, 180, 270]
split_list = ["train", "val", "test"]
for split in split_list:
    data_split_root = os.path.join(data_root, split)
    all_data_path = os.listdir(data_split_root)
    for data_path in all_data_path:
        now_path = os.path.join(data_split_root, data_path)
        now_list = os.listdir(now_path)
        for now in now_list:
            im = Image.open(os.path.join(now_path, now))
            for rot in rot_list:
                result_path = os.path.join("{}_rot_{}".format(now_path, rot).replace("omniglot_single", "omniglot_rot"), now)
                result_path = Tools.new_dir(result_path)
                new_im = im.rotate(rot)
                new_im.save(result_path)
                pass
        pass
    pass
