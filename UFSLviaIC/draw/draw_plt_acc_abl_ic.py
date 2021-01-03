import os
from glob import glob
import matplotlib.pyplot as plt  # matplotlib.colors.BASE_COLORS
from alisuretool.Tools import Tools


txt_path = "/mnt/4T/ALISURE/ActiveLearning/UFSLviaIC/models_abl/ic_res_xx"
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

    txt_name = all_txt_name[index]

    head = "*" if txt_name[-3] == "head" else ""
    if txt_name[0] == "vgg16":
        d = int(txt_name[3])
        name = "VGG-16"
    elif txt_name[0] == "conv4":
        d = int(txt_name[2])
        name = "Conv-4"
    else:
        d = int(txt_name[2])
        if txt_name[0] == "res50":
            name = "Resnet-50"
        elif txt_name[0] == "res34":
            name = "Resnet-34"
        elif txt_name[0] == "res18":
            name = "Resnet-18"
        else:
            raise Exception("")
        name = name + head
        pass

    if name not in acc_result_dict:
        acc_result_dict[name] = {}
    acc_result_dict[name][d] = acc_dict
    pass



x_data = ["64", "128", '256', '512', '1024', '2048']
color = ["r", "g", "b", "k", "y", "c", "g", "m"]
for split in ["Train", "Val", "Test"]:
    plt.figure(figsize=(8, 6))

    handles1, handles2, labels1, labels2 = [], [], [], []
    for acc_index, acc_key in enumerate(acc_result_dict):
        now_acc = acc_result_dict[acc_key]
        if len(now_acc.keys()) < len(x_data) or "*" in acc_key:
            continue
        top_1 = [now_acc[int(x)][split][0] for x in x_data]
        top_5 = [now_acc[int(x)][split][1] for x in x_data]
        ln1, = plt.plot(x_data, top_1, color=color[acc_index], linewidth=2.0, linestyle='-')
        plt.scatter(x_data, top_1, s=20, color=color[acc_index])
        ln2, = plt.plot(x_data, top_5, color=color[acc_index], linewidth=2.0, linestyle=':')
        plt.scatter(x_data, top_5, s=20, color=color[acc_index])
        handles1.append(ln1)
        handles2.append(ln2)
        labels1.append("{} {} Top 1".format(acc_key, split))
        labels2.append("{} {} Top 5".format(acc_key, split))
        pass

    plt.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='lower center', ncol=2, fontsize=12)
    plt.grid(linestyle='--')
    plt.ylim(0.0, 1.0)
    plt.locator_params("y", nbins=10)
    plt.xlabel('Dimension', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.tick_params(labelsize=16)
    plt.subplots_adjust(top=0.96, bottom=0.10, left=0.12, right=0.98, hspace=0, wspace=0)

    plt.savefig(Tools.new_dir(os.path.join("plot", "ic", "abl_acc_ic_{}.pdf".format(split))))
    plt.show()
    pass
