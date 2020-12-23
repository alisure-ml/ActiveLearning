import os
import sys
import shutil
from tqdm import tqdm
import scipy.io as scio
from alisuretool.Tools import Tools


data_dir = "/mnt/4T/Data/data/UFSL/Cars"
car_anno_path = os.path.join(data_dir, "cars_annos.mat")


dataset_list = ['train', 'val', 'test']
annos = scio.loadmat(car_anno_path)


anno_dict = {}
for anno in annos["annotations"][0][0].base:
    filename = anno[0][0]
    x1, y1, x2, y2 = anno[1][0][0], anno[2][0][0], anno[3][0][0], anno[4][0][0]
    class_id, is_test = anno[5][0][0], anno[6][0][0]
    anno_tuple = (class_id, is_test, (x1, y1, x2, y2), filename)
    if class_id not in anno_dict:
        anno_dict[class_id] = []
    anno_dict[class_id].append(anno_tuple)
    pass


for index, class_id in tqdm(enumerate(list(anno_dict.keys()))):
    split = 0
    if index < 100:
        split = 0
    elif index < 136:
        split = 1
    else:
        split = 2
        pass
    for anno in anno_dict[class_id]:
        filename = anno[-1]
        src_filename = os.path.join(data_dir, filename)
        dst_filename = Tools.new_dir(os.path.join(data_dir, dataset_list[split],
                                                  str(class_id), os.path.basename(filename)))
        shutil.copy(src_filename, dst_filename)
    pass

