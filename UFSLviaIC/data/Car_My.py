import os
import sys
import shutil
from PIL import Image
from tqdm import tqdm
import scipy.io as scio
from alisuretool.Tools import Tools


data_dir = "/mnt/4T/Data/data/UFSL/Cars"
# data_dir = '/media/ubuntu/4T/ALISURE/Data/UFSL/Cars'
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


result_size = 256
is_png = True

# result_size = None
# is_png = False

for index, class_id in tqdm(enumerate(list(anno_dict.keys()))):
    split = index % 7
    if split in [0, 1, 2, 3]:
        split = 0
    elif split in [4, 5]:
        split = 2
    else:
        split = 1
        pass

    for anno in anno_dict[class_id]:
        filename = anno[-1]
        src_filename = os.path.join(data_dir, filename)
        basename = os.path.basename(filename)
        if result_size is None:
            result_filename = "{}.png".format(basename[:-4]) if is_png else basename
            dst_filename = Tools.new_dir(os.path.join(data_dir, dataset_list[split], str(class_id), result_filename))
            shutil.copy(src_filename, dst_filename)
        else:
            dst_filename = Tools.new_dir(os.path.join(data_dir, "{}_png_7".format(result_size),
                                                      dataset_list[split], str(class_id), basename))
            Image.open(src_filename).resize((result_size, result_size)).save(dst_filename)
        pass
    pass

