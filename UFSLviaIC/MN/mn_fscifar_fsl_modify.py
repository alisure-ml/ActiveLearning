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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from mn_tool_fsl_test import FSLTestTool
from mn_tool_net import MatchingNet, Normalize, RunnerTool, ResNet12Small


##############################################################################################################


class CIFARDataset(object):

    def __init__(self, data_list, num_way, num_shot, image_size=32):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        if Config.aug_name == 1:
            normalize = transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
            self.transform = transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
            self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        elif Config.aug_name == 2:
            normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
            self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            raise Exception(".................")

        pass

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def get_data_all(data_root):
        train_folder = os.path.join(data_root, "train")

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

        return data_train_list

    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, now_label, now_image_filename = now_label_image_tuple
        now_label_k_shot_image_tuple = random.sample(self.data_dict[now_label], self.num_shot)

        # 其他样本
        other_label = list(self.data_dict.keys())
        other_label.remove(now_label)
        other_label = random.sample(other_label, self.num_way - 1)
        other_label_k_shot_image_tuple_list = []
        for _label in other_label:
            other_label_k_shot_image_tuple = random.sample(self.data_dict[_label], self.num_shot)
            other_label_k_shot_image_tuple_list.extend(other_label_k_shot_image_tuple)
            pass

        # c_way_k_shot
        c_way_k_shot_tuple_list = now_label_k_shot_image_tuple + other_label_k_shot_image_tuple_list
        random.shuffle(c_way_k_shot_tuple_list)

        task_list = c_way_k_shot_tuple_list + [now_label_image_tuple]
        task_data = torch.cat([torch.unsqueeze(self.read_image(one[2], self.transform), dim=0) for one in task_list])
        task_label = torch.Tensor([int(one_tuple[1] == now_label) for one_tuple in c_way_k_shot_tuple_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # all data
        self.data_train = CIFARDataset.get_data_all(Config.data_root)
        self.task_train = CIFARDataset(self.data_train, Config.num_way, Config.num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))
        self.norm = Normalize(2)

        # loss
        self.loss = RunnerTool.to_cuda(nn.MSELoss())

        # optim
        self.matching_net_optim = torch.optim.Adam(self.matching_net.parameters(), lr=Config.learning_rate)
        self.matching_net_scheduler = MultiStepLR(self.matching_net_optim, Config.train_epoch_lr, gamma=0.5)

        self.test_tool = FSLTestTool(self.matching_test, data_root=Config.data_root,
                                     num_way=Config.num_way_test, num_shot=Config.num_shot,
                                     episode_size=Config.episode_size, test_episode=Config.test_episode,
                                     transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load proto net success from {}".format(Config.mn_dir), txt_path=Config.log_file)
        pass

    def matching(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.matching_net(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        # 特征
        z_support, z_query = z.split(Config.num_shot * Config.num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, Config.num_way * Config.num_shot, z_dim)
        z_query_expand = z_query.expand(z_batch_size, Config.num_way * Config.num_shot, z_dim)

        # 相似性
        z_support = self.norm(z_support)
        similarities = torch.sum(z_support * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(z_batch_size, Config.num_way, Config.num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def matching_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)  # 5x64*5*5
        batch_z = self.matching_net(batches)  # 75x64*5*5
        z_support = sample_z.view(Config.num_way_test * Config.num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, Config.num_way_test * Config.num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, Config.num_way_test * Config.num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, Config.num_way_test, Config.num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def train(self):
        Tools.print()
        Tools.print("Training...", txt_path=Config.log_file)

        for epoch in range(1, 1 + Config.train_epoch):
            self.matching_net.train()

            Tools.print()
            all_loss = 0.0
            for task_data, task_labels, task_index in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                # 1 calculate features
                predicts = self.matching(task_data)

                # 2 loss
                loss = self.loss(predicts, task_labels)
                all_loss += loss.item()

                # 3 backward
                self.matching_net.zero_grad()
                loss.backward()
                self.matching_net_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} lr:{}".format(epoch, all_loss / len(self.task_train_loader),
                                                        self.matching_net_scheduler.get_last_lr()), txt_path=Config.log_file)

            self.matching_net_scheduler.step()
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                Tools.print()
                Tools.print("Test {} {} .......".format(epoch, Config.model_name), txt_path=Config.log_file)
                self.matching_net.eval()

                val_accuracy = self.test_tool.val(episode=epoch, is_print=True, has_test=False)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.matching_net.state_dict(), Config.mn_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch), txt_path=Config.log_file)
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################


class Config(object):
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    learning_rate = 0.001
    num_workers = 16
    # num_way = 5
    # num_way_test = 5
    # val_freq = 10
    num_shot = 1
    episode_size = 15
    test_episode = 600

    ##############################################################################################################
    dataset_name = "CIFARFS"
    # dataset_name = "FC100"

    if dataset_name == "CIFARFS":
        is_large = True
        # matching_net, net_name, batch_size = MatchingNet(hid_dim=64, z_dim=64), "conv4", 64
        matching_net = ResNet12Small(avg_pool=True, drop_rate=0.1, large=is_large)
        net_name, batch_size = "res12{}".format("large" if is_large else ""), 64
        aug_name = 1  # other
        train_epoch = 400
        train_epoch_lr = [200, 300]
        val_freq = 10
        num_way = 5
        num_way_test = 5
    else:
        # matching_net, net_name, batch_size = MatchingNet(hid_dim=64, z_dim=64), "conv4", 64
        matching_net, net_name, batch_size = ResNet12Small(avg_pool=True, drop_rate=0.1), "res12", 64
        aug_name = 1  # other
        # aug_name = 2  # my
        train_epoch = 60
        train_epoch_lr = [30, 50]
        val_freq = 2
        # num_way = 20
        num_way = 5
        num_way_test = 5
        pass
    ##############################################################################################################

    model_name = "{}_{}_{}_{}_{}_{}_{}_aug{}_{}".format(
        gpu_id, dataset_name, 32, net_name, train_epoch, num_way, num_shot, aug_name, val_freq)

    mn_dir = Tools.new_dir("../models_CIFARFS/mn/fsl_modify/{}.pkl".format(model_name))
    log_file = mn_dir.replace(".pkl", ".txt")
    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/{}'.format(dataset_name)
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/{}'.format(dataset_name)
    else:
        data_root = "F:\\data\\{}".format(dataset_name)

    Tools.print(model_name, txt_path=log_file)
    Tools.print(data_root, txt_path=log_file)
    Tools.print(mn_dir, txt_path=log_file)
    pass


##############################################################################################################


"""
2021-01-10 02:33:41 load proto net success from ../models_CIFARFS/mn/fsl_modify/2_CIFARFS_32_conv4_400_5_1_aug2.pkl
2021-01-10 02:33:56 Train 400 Accuracy: 0.7832222222222224
2021-01-10 02:34:10 Val   400 Accuracy: 0.5287777777777778
2021-01-10 02:34:24 Test1 400 Accuracy: 0.6115555555555555
2021-01-10 02:35:01 Test2 400 Accuracy: 0.6001333333333333
2021-01-10 02:38:07 episode=400, Test accuracy=0.6073111111111111
2021-01-10 02:38:07 episode=400, Test accuracy=0.6013555555555555
2021-01-10 02:38:07 episode=400, Test accuracy=0.6004222222222222
2021-01-10 02:38:07 episode=400, Test accuracy=0.6028444444444444
2021-01-10 02:38:07 episode=400, Test accuracy=0.6017333333333332
2021-01-10 02:38:07 episode=400, Mean Test accuracy=0.6027333333333333

2021-01-11 18:56:17 load proto net success from ../models_CIFARFS/mn/fsl_modify/3_CIFARFS_32_conv4_400_5_1_aug1.pkl
2021-01-11 18:56:37 Train 400 Accuracy: 0.8556666666666668
2021-01-11 18:56:55 Val   400 Accuracy: 0.56
2021-01-11 18:57:15 Test1 400 Accuracy: 0.6452222222222223
2021-01-11 18:58:02 Test2 400 Accuracy: 0.6318444444444444
2021-01-11 19:01:44 episode=400, Test accuracy=0.6412666666666667
2021-01-11 19:01:44 episode=400, Test accuracy=0.6355333333333333
2021-01-11 19:01:44 episode=400, Test accuracy=0.6336444444444445
2021-01-11 19:01:44 episode=400, Test accuracy=0.6380444444444445
2021-01-11 19:01:44 episode=400, Test accuracy=0.6370666666666667
2021-01-11 19:01:44 episode=400, Mean Test accuracy=0.6371111111111112


2021-01-10 21:04:30 load proto net success from ../models_CIFARFS/mn/fsl_modify/2_CIFARFS_32_res12_400_5_1_aug2.pkl
2021-01-10 21:04:51 Train 400 Accuracy: 0.9308888888888889
2021-01-10 21:05:14 Val   400 Accuracy: 0.5978888888888889
2021-01-10 21:05:35 Test1 400 Accuracy: 0.6652222222222222
2021-01-10 21:06:39 Test2 400 Accuracy: 0.6587111111111111
2021-01-10 21:11:36 episode=400, Test accuracy=0.6641333333333334
2021-01-10 21:11:36 episode=400, Test accuracy=0.6618666666666667
2021-01-10 21:11:36 episode=400, Test accuracy=0.6624
2021-01-10 21:11:36 episode=400, Test accuracy=0.663688888888889
2021-01-10 21:11:36 episode=400, Test accuracy=0.6625333333333334
2021-01-10 21:11:36 episode=400, Mean Test accuracy=0.6629244444444444

2021-01-11 15:32:58 load proto net success from ../models_CIFARFS/mn/fsl_modify/2_CIFARFS_32_res12_400_5_1_aug1.pkl
2021-01-11 15:33:17 Train 400 Accuracy: 0.98
2021-01-11 15:33:36 Val   400 Accuracy: 0.5976666666666667
2021-01-11 15:33:55 Test1 400 Accuracy: 0.6706666666666666
2021-01-11 15:34:47 Test2 400 Accuracy: 0.6677333333333333
2021-01-11 15:40:31 episode=400, Test accuracy=0.6790666666666667
2021-01-11 15:40:31 episode=400, Test accuracy=0.6810222222222222
2021-01-11 15:40:31 episode=400, Test accuracy=0.6826666666666668
2021-01-11 15:40:31 episode=400, Test accuracy=0.6745777777777778
2021-01-11 15:40:31 episode=400, Test accuracy=0.6694888888888889
2021-01-11 15:40:31 episode=400, Mean Test accuracy=0.6773644444444445
"""


"""
2021-01-11 14:35:22 load proto net success from ../models_CIFARFS/mn/fsl_modify/3_FC100_32_conv4_400_5_1_aug1.pkl
2021-01-11 14:35:36 Train 400 Accuracy: 0.8192222222222222
2021-01-11 14:35:51 Val   400 Accuracy: 0.3088888888888889
2021-01-11 14:36:05 Test1 400 Accuracy: 0.3447777777777778
2021-01-11 14:36:44 Test2 400 Accuracy: 0.35035555555555553
2021-01-11 14:39:55 episode=400, Test accuracy=0.3497777777777778
2021-01-11 14:39:55 episode=400, Test accuracy=0.3494888888888889
2021-01-11 14:39:55 episode=400, Test accuracy=0.34575555555555554
2021-01-11 14:39:55 episode=400, Test accuracy=0.3510888888888889
2021-01-11 14:39:55 episode=400, Test accuracy=0.3526222222222222
2021-01-11 14:39:55 episode=400, Mean Test accuracy=0.3497466666666667

2021-01-11 14:44:22 load proto net success from ../models_CIFARFS/mn/fsl_modify/2_FC100_32_res12_400_5_1_aug1.pkl
2021-01-11 14:44:47 Train 400 Accuracy: 0.9253333333333335
2021-01-11 14:45:12 Val   400 Accuracy: 0.3194444444444445
2021-01-11 14:45:38 Test1 400 Accuracy: 0.383
2021-01-11 14:47:10 Test2 400 Accuracy: 0.37133333333333335
2021-01-11 14:54:45 episode=400, Test accuracy=0.37853333333333333
2021-01-11 14:54:45 episode=400, Test accuracy=0.37202222222222225
2021-01-11 14:54:45 episode=400, Test accuracy=0.3735111111111111
2021-01-11 14:54:45 episode=400, Test accuracy=0.3749111111111111
2021-01-11 14:54:45 episode=400, Test accuracy=0.3738444444444445
2021-01-11 14:54:45 episode=400, Mean Test accuracy=0.3745644444444444
"""


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    runner.matching_net.eval()
    runner.test_tool.val(episode=0, is_print=True)

    runner.train()

    runner.load_model()
    runner.matching_net.eval()
    runner.test_tool.val(episode=Config.train_epoch, is_print=True, txt_path=Config.log_file)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True, txt_path=Config.log_file)
    pass
