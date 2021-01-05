import os
import time
from glob import glob
import matplotlib.pyplot as plt
# import matplotlib.colors.BASE_COLORS
from alisuretool.Tools import Tools


dataset_name = "miniimagenet"

txt_path = "/mnt/4T/ALISURE/ActiveLearning/UFSLviaIC/models_abl/{}/mn/result".format(dataset_name)
if not os.path.exists(txt_path):
    txt_path = "/media/ubuntu/4T/ALISURE/ActiveLearning/UFSLviaIC/models_abl/{}/mn/result".format(dataset_name)
    pass

all_txt = glob(os.path.join(txt_path, "*.txt"))
all_txt_name = []
all_txt_content = []
for txt in all_txt:
    with open(txt) as file:
        txt_content = file.readlines()
        all_txt_content.append(txt_content)
        pass
    txt_name_list = os.path.basename(txt).split("_")[:-1]
    all_txt_name.append(txt_name_list)
    pass


ways_shots_dict = {}
for index, txt_content in enumerate(all_txt_content):
    txt_content = [one for one in txt_content if "acc=" in one]
    acc_result = [[split_one.split("=")[1].strip() for split_one in one.split("w")[1].split(",")] for one in txt_content]

    ways_dict = {}
    shots_dict = {}
    for acc_index, acc_one in enumerate(acc_result):
        way, shot = int(acc_one[0]), int(acc_one[1])
        result = {"acc": float(acc_one[2]), "con": float(acc_one[3])}
        if way == 5 and acc_index > 5:
            shots_dict[shot] = result
        else:
            ways_dict[way] = result
        pass

    txt_name = all_txt_name[index]
    if txt_name[0] == "ufsl":
        name = "Our"
    elif txt_name[0] == "cluster":
        name = "Cluster"
    elif txt_name[0] == "css":
        name = "CSS"
    elif txt_name[0] == "random":
        name = "Random"
        pass

    ways_shots_dict["-".join(txt_name)] = {"ways": ways_dict, "shots": shots_dict}
    pass


color = ["r", "g", "b", "k", "y", "c", "m"]
linestyle = ["-", ":"]
for split in ["shots", "ways"]:
    plt.figure(figsize=(8, 6))

    handles1, labels1 = [], []
    for acc_index, key in enumerate(ways_shots_dict):
        ways_and_shots = ways_shots_dict[key]
        nows = ways_and_shots[split]
        now_keys = sorted(list(nows.keys()))
        now_values = [nows[key]["acc"] for key in now_keys]
        now_keys = [str(key) for key in now_keys]

        ln1, = plt.plot(now_keys, now_values, color=color[acc_index % len(color)],
                        linewidth=2.0, linestyle=linestyle[acc_index // len(color)])
        plt.scatter(now_keys, now_values, s=20, color=color[acc_index % len(color)])
        handles1.append(ln1)
        labels1.append(key)
        pass

    plt.legend(handles=handles1, labels=labels1, loc='best', ncol=2, fontsize=14)
    plt.grid(linestyle='--')
    plt.ylim(0.0, 0.9)
    plt.locator_params("y", nbins=10)
    plt.tick_params(labelsize=16)
    # plt.subplots_adjust(top=0.96, bottom=0.10, left=0.12, right=0.98, hspace=0, wspace=0)
    plt.subplots_adjust(top=0.96, bottom=0.06, left=0.09, right=0.98, hspace=0, wspace=0)

    plt.savefig(Tools.new_dir(os.path.join("plot", "fsl", "{}_{}.pdf".format(dataset_name, split))))
    plt.show()
    pass
