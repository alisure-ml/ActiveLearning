import os
from glob import glob
from PIL import Image
from collections import Counter


class Config(object):
    vis_root = "../vis/miniImagenet/ic_res_xx/3_resnet_34_64_512_1_2100_500_200"

    split = "train"
    # split = "val"
    # split = "test"
    vis_ic_path = os.path.join(vis_root, split)

    ic_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ic_id_list += [i + 100 for i in ic_id_list]
    ic_id_list += [i + 200 for i in ic_id_list]
    image_size = 84
    image_num = 50
    margin = 4
    result_size = (image_num * image_size + (image_num - 1) * margin,
                   len(ic_id_list) * image_size + (len(ic_id_list) - 1) * margin)

    result_path = os.path.join(vis_root, "{}_{}_{}_{}.png".format(split, image_size, image_num, margin))
    pass


def get_image_by_id(ic_id):
    ic_image_file = glob(os.path.join(Config.vis_ic_path, str(ic_id), "*.png"))
    ic_class = [os.path.basename(ic).split("_")[0] for ic in ic_image_file]
    ic_count_sorted = sorted(Counter(ic_class).items(), key=lambda x: x[1], reverse=True)

    ic_image_file_sorted = []
    for ic_count_one in ic_count_sorted:
        ic_image_file_sorted.extend(glob(os.path.join(Config.vis_ic_path, str(ic_id),
                                                      "{}_*.png".format(ic_count_one[0]))))
        pass

    im_list = []
    for image_file in ic_image_file_sorted:
        im = Image.open(image_file)
        im_list.append(im)
        pass
    return im_list


if __name__ == '__main__':
    im_result = Image.new("RGB", size=Config.result_size, color=(255, 255, 255))
    for i in range(len(Config.ic_id_list)):
        im_list = get_image_by_id(Config.ic_id_list[i])
        image_num = Config.image_num if len(im_list) > Config.image_num else len(im_list)
        for j in range(image_num):
            now_ic = im_list[j]
            im_result.paste(now_ic.resize((Config.image_size, Config.image_size)),
                            box=(j * (Config.image_size + Config.margin), i * (Config.image_size + Config.margin)))
        pass
    im_result.save(Config.result_path)
    pass
