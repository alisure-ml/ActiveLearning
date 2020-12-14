# https://github.com/nupurkmr9/S2M2_fewshot/blob/master/filelists/CUB/make_json.py
import os
import json
import random
import numpy as np


data_dir = "/mnt/4T/Data/data/CUB_200_2011"
data_path = os.path.join(data_dir, 'images')
dataset_list = ['base', 'val', 'novel']

folder_list = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list, range(0, len(folder_list))))

class_file_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = os.path.join(data_path, folder)
    class_file_list_all.append([os.path.join(folder_path, cf) for cf in os.listdir(folder_path)
                                if (os.path.isfile(os.path.join(folder_path, cf)) and cf[0] != '.')])
    random.shuffle(class_file_list_all[i])
    pass

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, class_file_list in enumerate(class_file_list_all):
        if 'base' in dataset:
            if i % 2 == 0:
                file_list = file_list + class_file_list
                label_list = label_list + np.repeat(i, len(class_file_list)).tolist()
        if 'val' in dataset:
            if i % 4 == 1:
                file_list = file_list + class_file_list
                label_list = label_list + np.repeat(i, len(class_file_list)).tolist()
        if 'novel' in dataset:
            if i % 4 == 3:
                file_list = file_list + class_file_list
                label_list = label_list + np.repeat(i, len(class_file_list)).tolist()
        pass

    fo = open(os.path.join(data_dir, "{}.json".format(dataset)), "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" % dataset)
    pass
