import os
import json
import random
import shutil
import numpy as np
from alisuretool.Tools import Tools


data_dir = "/mnt/4T/Data/data/CUB_200_2011/images"
dataset_list = ['train', 'val', 'test']
result_dir = Tools.new_dir("/mnt/4T/Data/data/UFSL/CUB")

folder_list = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
folder_list.sort()

for i, folder in enumerate(folder_list):
    Tools.print("{}/{} {}".format(i, len(folder_list), folder))
    split = "train"
    if i % 2 == 0:
        split = "train"
    elif i % 4 == 1:
        split = "val"
    elif i % 4 == 3:
        split = "test"
    else:
        Tools.print("..........")

    folder_path = os.path.join(data_dir, folder)
    for cf in os.listdir(folder_path):
        now_path = os.path.join(folder_path, cf)
        result_path = Tools.new_dir(os.path.join(result_dir, split, folder, cf))
        if os.path.isfile(now_path) and cf[0] != '.':
            shutil.copy(now_path, result_path)
        pass
    pass
