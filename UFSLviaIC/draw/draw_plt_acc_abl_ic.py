import os
import time
from glob import glob
import matplotlib.pyplot as plt  # matplotlib.colors.BASE_COLORS
from alisuretool.Tools import Tools


# txt_path = "/mnt/4T/ALISURE/ActiveLearning/UFSLviaIC/models_abl/ic_res_xx_small"
txt_path = "/media/ubuntu/4T/ALISURE/ActiveLearning/UFSLviaIC/models_abl/ic_res_xx_small"

all_txt = glob(os.path.join(txt_path, "*.txt"))
all_txt_name = []
all_txt_content = []
for txt in all_txt:
    with open(txt) as file:
        txt_content = file.readlines()
        all_txt_content.append(txt_content)
        pass
    txt_name_list = os.path.basename(txt).split("_")[2:]
    all_txt_name.append(txt_name_list)
    pass


acc_result_dict = {}
for index, txt_content in enumerate(all_txt_content):
    acc_dict = {}
    for split in ["Train", "Val", "Test"]:
        txt_acc = [txt.split(" ")[-2].split("/") for txt in txt_content if "Epoch" in txt and split in txt]
        txt_acc = [[float(acc[0]), float(acc[1])] for acc in txt_acc]
        top_1, top_5 = txt_acc[-1]
        acc_dict[split] = [top_1, top_5]
        pass

    time_list = [time.strptime(txt.split(" E")[0], "%Y-%m-%d %H:%M:%S")
                 for txt in txt_content if "Epoch" in txt and "ic_lr" in txt][100: 110]
    now_time = (time.mktime(time_list[-1]) - time.mktime(time_list[0])) // (len(time_list) - 1)

    txt_name = all_txt_name[index]

    if txt_name[0] == "vgg16":
        d = int(txt_name[3])
        name = "VGG-16"
    elif txt_name[0] == "conv4":
        d = int(txt_name[2])
        name = "Conv-4"
    else:
        d = int(txt_name[2])
        if txt_name[0] == "res50":
            name = "ResNet-50"
        elif txt_name[0] == "res34":
            name = "ResNet-34"
        elif txt_name[0] == "res18":
            name = "ResNet-18"
        else:
            raise Exception("")
        pass

    if name not in acc_result_dict:
        acc_result_dict[name] = {}
    acc_result_dict[name][d] = {"acc": acc_dict, "time": now_time}
    pass


x_data = ["64", "128", '256', '512', '1024', '2048']
color = ["g", "b", "r", "k", "y", "c", "g", "m"]
for split in ["Train", "Val", "Test"]:
    plt.figure(figsize=(8, 6))

    handles1, handles2, labels1, labels2 = [], [], [], []
    acc_result_key = sorted(list(acc_result_dict.keys()))
    for acc_index, acc_key in enumerate(acc_result_key):
        now_acc = acc_result_dict[acc_key]
        if len(now_acc.keys()) < len(x_data) or "*" in acc_key:
            continue
        top_1 = [now_acc[int(x)]["acc"][split][0] for x in x_data]
        top_5 = [now_acc[int(x)]["acc"][split][1] for x in x_data]
        ln1, = plt.plot(x_data, top_1, color=color[acc_index], linewidth=2.0, linestyle='-')
        plt.scatter(x_data, top_1, s=20, color=color[acc_index])
        # ln2, = plt.plot(x_data, top_5, color=color[acc_index], linewidth=2.0, linestyle='-.')
        # plt.scatter(x_data, top_5, s=20, color=color[acc_index])
        handles1.append(ln1)
        # handles2.append(ln2)
        # labels1.append("{} top-1".format(acc_key))
        # labels2.append("{} top-5".format(acc_key))
        labels1.append("{}".format(acc_key))
        pass

    plt.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='best', ncol=1, fontsize=14)
    plt.grid(linestyle='--')
    plt.ylim(0.35, 0.7)
    plt.locator_params("y", nbins=10)
    plt.xlabel('dimension', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    plt.tick_params(labelsize=16)
    plt.subplots_adjust(top=0.96, bottom=0.10, left=0.12, right=0.99, hspace=0, wspace=0)

    plt.savefig(Tools.new_dir(os.path.join("plot_new", "ic", "abl_acc_ic_{}2.pdf".format(split))))
    plt.show()
    pass


x_data = ["64", "128", '256', '512', '1024', '2048']
color = ["g", "b", "r", "k", "y", "c", "g", "m"]
plt.figure(figsize=(8, 6))

handles1, labels1 = [], []
acc_result_key = sorted(list(acc_result_dict.keys()))
for acc_index, acc_key in enumerate(acc_result_key):
    if "*" in acc_key:
        continue
    now_acc = acc_result_dict[acc_key]

    time = [now_acc[int(x)]["time"] for x in x_data]
    ln1, = plt.plot(x_data, time, color=color[acc_index], linewidth=2.0, linestyle='-')
    plt.scatter(x_data, time, s=20, color=color[acc_index])
    handles1.append(ln1)
    labels1.append("{}".format(acc_key))
    pass

plt.legend(handles=handles1, labels=labels1, loc='best', ncol=1, fontsize=14)
plt.grid(linestyle='--')
plt.ylim(0, 90)
plt.locator_params("y", nbins=10)
plt.xlabel('dimension', fontsize=18)
plt.ylabel('training time per epoch (in seconds)', fontsize=18)
plt.tick_params(labelsize=16)
plt.subplots_adjust(top=0.96, bottom=0.10, left=0.10, right=0.99, hspace=0, wspace=0)

plt.savefig(Tools.new_dir(os.path.join("plot_new", "ic", "abl_acc_ic_Train_time2.pdf")))
plt.show()

