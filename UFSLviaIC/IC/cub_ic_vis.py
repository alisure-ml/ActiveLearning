import os
import math
import torch
import random
import platform
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from alisuretool.Tools import Tools
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50, vgg16_bn


##############################################################################################################


class CUBIC(Dataset):

    def __init__(self, data_list, image_size=84):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                                                  transforms.CenterCrop(image_size), transforms.ToTensor(), normalize])
        self.transform2 = transforms.Compose([transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                                              transforms.CenterCrop(image_size), transforms.ToTensor()])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image_transform = self.transform(image)
        image_transform2 = self.transform2(image)
        return image_transform, image_transform2, label, idx

    @staticmethod
    def get_data_all(data_root):
        train_folder = os.path.join(data_root, "train")
        val_folder = os.path.join(data_root, "val")
        test_folder = os.path.join(data_root, "test")

        count_image, count_class, data_train_list = 0, 0, []
        for label in os.listdir(train_folder):
            now_class_path = os.path.join(train_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_train_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        count_image, count_class, data_val_list = 0, 0, []
        for label in os.listdir(val_folder):
            now_class_path = os.path.join(val_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_val_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        count_image, count_class, data_test_list = 0, 0, []
        for label in os.listdir(test_folder):
            now_class_path = os.path.join(test_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_test_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        return data_train_list, data_val_list, data_test_list

    pass


##############################################################################################################


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ICResNet(nn.Module):

    def __init__(self, low_dim=512, modify_head=False, resnet=None, vggnet=None):
        super().__init__()
        self.is_res = True if resnet else False
        self.is_vgg = True if vggnet else False

        if self.is_res:
            self.resnet = resnet(num_classes=low_dim)
            if modify_head:
                self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                pass
        elif self.is_vgg:
            self.vggnet = vggnet()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, low_dim)
            pass
        else:
            raise Exception("......")

        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        if self.is_res:
            out_logits = self.resnet(x)
        elif self.is_vgg:
            features = self.vggnet.features(x)
            features = self.avgpool(features)
            features = torch.flatten(features, 1)
            out_logits = self.fc(features)
            pass
        else:
            raise Exception("......")

        out_l2norm = self.l2norm(out_logits)
        return out_logits, out_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        # data
        self.data_train, self.data_val, self.data_test = CUBIC.get_data_all(Config.data_root)
        self.train_loader = DataLoader(CUBIC(self.data_train), Config.batch_size, False, num_workers=Config.num_workers)
        self.val_loader = DataLoader(CUBIC(self.data_val), Config.batch_size, False, num_workers=Config.num_workers)
        self.test_loader = DataLoader(CUBIC(self.data_test), Config.batch_size, False, num_workers=Config.num_workers)

        # model
        self.ic_model = self.to_cuda(ICResNet(Config.ic_out_dim, modify_head=Config.modify_head,
                                              resnet=Config.resnet, vggnet=Config.vggnet))
        pass

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    def load_model(self):
        if os.path.exists(Config.ic_dir):
            self.ic_model.load_state_dict(torch.load(Config.ic_dir))
            Tools.print("load ic model success from {}".format(Config.ic_dir))
        pass

    def vis(self, split="train"):
        Tools.print()
        Tools.print("Vis ...")

        loader = self.test_loader if split == "test" else self.train_loader
        loader = self.val_loader if split == "val" else loader

        feature_list = []
        self.ic_model.eval()
        for image_transform, image, label, idx in tqdm(loader):
            ic_out_logits, ic_out_l2norm = self.ic_model(self.to_cuda(image_transform))

            image_data = np.asarray(image.permute(0, 2, 3, 1) * 255, np.uint8)
            cluster_id = np.asarray(torch.argmax(ic_out_logits, -1).cpu())
            for i in range(len(idx)):
                feature_list.append([int(idx[i]), int(label[i]), int(cluster_id[i]),
                                     np.array(ic_out_logits[i].cpu().detach().numpy()),
                                     np.array(ic_out_l2norm[i].cpu().detach().numpy())])

                result_path = Tools.new_dir(os.path.join(Config.vis_dir, split, str(cluster_id[i])))
                Image.fromarray(image_data[i]).save(os.path.join(result_path, "{}_{}.png".format(label[i], idx[i])))
                pass
            pass

        Tools.write_to_pkl(os.path.join(Config.vis_dir, "{}.pkl".format(split)), feature_list)
        pass

    pass


##############################################################################################################


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8
    batch_size = 8

    resnet, vggnet, net_name = resnet18, None, "resnet_18"
    # resnet, vggnet, net_name = resnet34, None, "resnet_34"
    # resnet, vggnet, net_name = resnet50, None, "resnet_50"
    # resnet, vggnet, net_name = None, vgg16_bn, "vgg16_bn"

    modify_head = False
    # modify_head = True

    # ic
    ic_out_dim = 512

    ic_dir = "../cub/models/ic_res_xx/2_resnet_18_64_512_1_2100_500_200_False_ic.pkl"

    vis_dir = Tools.new_dir("../vis/CUB/ic_res_xx/2_resnet_18_64_512_1_2100_500_200_False_ic")

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/CUB'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/CUB'
    else:
        data_root = "F:\\data\\CUB"
    Tools.print(data_root)
    pass


if __name__ == '__main__':
    runner = Runner()
    runner.load_model()

    runner.vis(split="train")
    runner.vis(split="val")
    runner.vis(split="test")

    pass
