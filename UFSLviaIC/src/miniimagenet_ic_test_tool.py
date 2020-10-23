import os
import torch
import numpy as np
from PIL import Image
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


##############################################################################################################


class MiniImageNetIC(Dataset):

    def __init__(self, data_list, image_size=84):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        norm = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                    std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.transform = transforms.Compose([transforms.CenterCrop(size=image_size), transforms.ToTensor(), norm])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = image if self.transform is None else self.transform(image)
        return image, label, idx

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


class KNN(object):

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    @classmethod
    def cal(cls, labels, dist, train_labels, max_c, k, t):
        # ---------------------------------------------------------------------------------- #
        batch_size = labels.size(0)
        yd, yi = dist.topk(k + 1, dim=1, largest=True, sorted=True)
        yd, yi = yd[:, 1:], yi[:, 1:]
        retrieval = train_labels[yi]

        retrieval_1_hot = cls.to_cuda(torch.zeros(k, max_c)).resize_(batch_size * k, max_c).zero_().scatter_(
            1, retrieval.view(-1, 1), 1).view(batch_size, -1, max_c)
        yd_transform = yd.clone().div_(t).exp_().view(batch_size, -1, 1)
        probs = torch.sum(torch.mul(retrieval_1_hot, yd_transform), 1)
        _, predictions = probs.sort(1, True)
        # ---------------------------------------------------------------------------------- #

        correct = predictions.eq(labels.data.view(-1, 1))

        top1 = correct.narrow(1, 0, 1).sum().item()
        top5 = correct.narrow(1, 0, 5).sum().item()
        return top1, top5

    @classmethod
    def knn(cls, feature_encoder, ic_model, low_dim, train_loader, k, t=0.1):

        with torch.no_grad():
            n_sample = train_loader.dataset.__len__()
            out_memory = cls.to_cuda(torch.zeros(n_sample, low_dim).t())
            train_labels = cls.to_cuda(torch.LongTensor(train_loader.dataset.train_label))
            max_c = train_labels.max() + 1

            out_list = []
            for batch_idx, (inputs, labels, indexes) in enumerate(train_loader):
                inputs = cls.to_cuda(inputs)

                if feature_encoder is None:
                    _, out_l2norm = ic_model(inputs)
                else:
                    features = feature_encoder(inputs)  # 5x64*19*19
                    _, out_l2norm = ic_model(features)
                    pass

                out_list.append([out_l2norm, cls.to_cuda(labels)])
                out_memory[:, batch_idx * inputs.size(0):(batch_idx + 1) * inputs.size(0)] = out_l2norm.data.t()
                pass

            top1, top5, total = 0., 0., 0
            for out in out_list:
                dist = torch.mm(out[0], out_memory)
                _top1, _top5 = cls.cal(out[1], dist, train_labels, max_c, k, t)

                top1 += _top1
                top5 += _top5
                total += out[1].size(0)
                pass

            return top1 / total, top5 / total

        pass

    pass


##############################################################################################################


class ICTestTool(object):

    def __init__(self, feature_encoder, ic_model, data_root, batch_size=64, num_workers=8, ic_out_dim=512):
        self.feature_encoder = feature_encoder if feature_encoder is None else self.to_cuda(feature_encoder)
        self.ic_model = self.to_cuda(ic_model)
        self.ic_out_dim = ic_out_dim

        # data
        self.data_train, self.data_val, self.data_test = MiniImageNetIC.get_data_all(data_root)
        self.train_loader = DataLoader(MiniImageNetIC(self.data_train), batch_size, False, num_workers=num_workers)
        self.val_loader = DataLoader(MiniImageNetIC(self.data_val), batch_size, False, num_workers=num_workers)
        self.test_loader = DataLoader(MiniImageNetIC(self.data_test), batch_size, False, num_workers=num_workers)
        pass

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    def val_ic(self, ic_loader):
        acc_1, acc_2 = KNN.knn(self.feature_encoder, self.ic_model, self.ic_out_dim, ic_loader, 100)
        return acc_1, acc_2

    def val(self, epoch, is_print=True):
        if is_print:
            Tools.print()
            Tools.print("Test {} .......".format(epoch))
            pass

        acc_1_train, acc_2_train = self.val_ic(ic_loader=self.train_loader)
        acc_1_val, acc_2_val = self.val_ic(ic_loader=self.val_loader)
        acc_1_test, acc_2_test = self.val_ic(ic_loader=self.test_loader)

        if is_print:
            Tools.print("Epoch: [{}] Train {:.4f}/{:.4f}".format(epoch, acc_1_train, acc_2_train))
            Tools.print("Epoch: [{}] Val   {:.4f}/{:.4f}".format(epoch, acc_1_val, acc_2_val))
            Tools.print("Epoch: [{}] Test  {:.4f}/{:.4f}".format(epoch, acc_1_test, acc_2_test))
            pass
        return acc_1_val

    pass


if __name__ == '__main__':
    ic_test_tool = ICTestTool(feature_encoder=None, ic_model=None,
                              data_root=None, batch_size=64, num_workers=8, ic_out_dim=512)
    ic_test_tool.val(0)
    pass
