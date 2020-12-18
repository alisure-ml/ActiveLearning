import os
import numpy as np
from glob import glob
from tqdm import tqdm
from  PIL import Image
from skimage import measure
from alisuretool.Tools import Tools


data_dir = "/mnt/4T/Data/data/UFSL/CUB"
dataset_list = ['train', 'val', 'test']
seg_str = "segmentations"
result_dir = Tools.new_dir(os.path.join(data_dir, "CUBSeg"))

for i, split in enumerate(dataset_list):
    Tools.print("{}/{} {}".format(i, len(dataset_list), split))

    now_dir = os.path.join(data_dir, split)
    all_image = glob(os.path.join(now_dir, "*/*.jpg"))
    for image_one in tqdm(all_image):
        seg_file = image_one.replace(split, seg_str).replace(".jpg", ".png")
        seg_im = Image.open(seg_file)
        jpg_im = Image.open(image_one)

        seg_2 = np.asarray(seg_im) > 0
        seg_label_image = measure.label(seg_2)
        seg_region_list = measure.regionprops(seg_label_image)

        max_area = np.argmax([_.area for _ in seg_region_list])
        seg_region = seg_region_list[max_area]

        bbox = seg_region.bbox
        bbox = (bbox[1]-5, bbox[0]-5, bbox[3]+5, bbox[2]+5)
        result_filename = Tools.new_dir(os.path.join(result_dir, image_one.split("/CUB/")[1]))
        jpg_im.crop(bbox).save(result_filename)
        pass
    pass
