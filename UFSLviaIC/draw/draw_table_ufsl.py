import os
import time
from glob import glob
import matplotlib.pyplot as plt  # matplotlib.colors.BASE_COLORS
from alisuretool.Tools import Tools


# dataset_name = "miniimagenet"
dataset_name = "tieredimagenet"

txt_path = "/media/ubuntu/4T/ALISURE/ActiveLearning/UFSLviaIC/models_abl/{}/mn/result_table".format(dataset_name)
all_txt = glob(os.path.join(txt_path, "*.txt"))
all_txt_name = []
all_txt_content = []
for txt in all_txt:
    txt_name_list = os.path.basename(txt).split("_")
    if len(txt_name_list) == 5 and "18" in txt_name_list[2]:
        continue
    with open(txt) as file:
        txt_content = file.readlines()
        all_txt_content.append(txt_content)
        pass
    txt_name_list = [txt_name_list[1], txt_name_list[-2]]
    all_txt_name.append(txt_name_list)
    pass


acc_result_dict = {}
for index, txt_content in enumerate(all_txt_content):
    txt_name = all_txt_name[index]
    method_type, backbone = txt_name
    if method_type == "cluster":
        method_type = "Clustering"
    elif method_type == "css":
        method_type = "CSS"
    elif method_type == "ufsl":
        method_type = "Ours"
    elif method_type == "random":
        method_type = "Random"
    elif method_type == "label":
        method_type = "Label"

    if backbone == "conv4":
        backbone = "Conv-4"
    elif backbone == "res12":
        backbone = "ResNet-12"
        pass

    if method_type not in acc_result_dict:
        acc_result_dict[method_type] = {}

    acc_result_dict[method_type][backbone] = {}
    txt_content = [text for text in txt_content if "way" in text]
    txt_data = [[one.split("=")[1] for one in text.strip().split(" ")[-1].split(",")] for text in txt_content]
    for txt in txt_data:
        acc_result_dict[method_type][backbone]["{}-{}".format(txt[0], txt[1])] = {"acc": float(txt[2]) * 100,
                                                                                  "con": float(txt[3]) * 100}
    pass


# Random & Conv-4 & 32.5\tiny{$\pm$0.12\%} & 76.5\tiny{$\pm$0.12\%} & 32.5\tiny{$\pm$0.12\%} & 32.5\tiny{$\pm$0.12\%} \\
for backbone_key in ["Conv-4", "ResNet-12"]:
    result_str = []
    for method_key in ["Random", "CSS", "Clustering", "Label", "Ours"]:
        acc_dict = acc_result_dict[method_key][backbone_key]
        now_str = "{} & {} & {:.2f}\\small{} & {:.2f}\\small{} & {:.2f}\\small{} & {:.2f}\\small{} \\\\".format(
            method_key, backbone_key,
            acc_dict["5-1"]["acc"], "{$\\pm$" + "{:.2f}".format(acc_dict["5-1"]["con"]) + "\\%}",
            acc_dict["5-5"]["acc"], "{$\\pm$" + "{:.2f}".format(acc_dict["5-5"]["con"]) + "\\%}",
            acc_dict["20-1"]["acc"], "{$\\pm$" + "{:.2f}".format(acc_dict["20-1"]["con"]) + "\\%}",
            acc_dict["20-5"]["acc"], "{$\\pm$" + "{:.2f}".format(acc_dict["20-5"]["con"]) + "\\%}",
        )
        result_str.append(now_str)
        pass

    for i in result_str:
        print(i)
    Tools.print()
    pass

