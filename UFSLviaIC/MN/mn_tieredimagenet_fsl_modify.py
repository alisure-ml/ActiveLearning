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
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from mn_tool_fsl_test import FSLTestTool
from mn_tool_net import MatchingNet, Normalize, RunnerTool, ResNet12Small


##############################################################################################################


class TieredImageNetDataset(object):

    def __init__(self, data_list, num_way, num_shot, load_data=False):
        self.data_list, self.num_way, self.num_shot = data_list, num_way, num_shot
        self.load_data = load_data
        if self.load_data:
            # self.data_image = self.load_image_data()
            Tools.print("Load image to memory....")
            self.data_image_path = os.path.join(Config.data_root, "train_images_png.npz")
            self.data_image = np.load(self.data_image_path)["images"]
            pass

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                         std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.transform = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        pass

    def load_image_data(self):
        result = []
        for image_index, image_tuple in tqdm(enumerate(self.data_list), total=len(self.data_list)):
            index, class_id, image_path = image_tuple
            assert image_index == index
            image_data = np.asarray(Image.open(image_path))
            result.append(image_data)
            pass
        return result

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
        task_data = torch.cat([torch.unsqueeze(self.read_image(one, self.transform), dim=0) for one in task_list])
        task_label = torch.Tensor([int(one_tuple[1] == now_label) for one_tuple in c_way_k_shot_tuple_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index

    def read_image(self, one, transform=None):
        if self.load_data:
            image_id = int(os.path.basename(one[-1]).split(".png")[0])
            now_data = self.data_image[image_id]
            image = Image.fromarray(now_data)
        else:
            image = Image.open(one[2]).convert('RGB')

        if transform is not None:
            image = transform(image)
        return image

    pass


##############################################################################################################


class Runner(object):

    def __init__(self):
        self.best_accuracy = 0.0

        # all data
        self.data_train = TieredImageNetDataset.get_data_all(Config.data_root)
        self.task_train = TieredImageNetDataset(self.data_train, Config.num_way,
                                                Config.num_shot, load_data=Config.load_data)
        self.task_train_loader = DataLoader(self.task_train, Config.batch_size, True, num_workers=Config.num_workers)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.matching_net)
        self.matching_net = RunnerTool.to_cuda(nn.DataParallel(self.matching_net))
        cudnn.benchmark = True
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))
        self.norm = Normalize(2)

        # loss
        self.loss = RunnerTool.to_cuda(nn.MSELoss())

        # optim
        self.matching_net_optim = torch.optim.Adam(self.matching_net.parameters(), lr=Config.learning_rate)
        self.matching_net_scheduler = MultiStepLR(self.matching_net_optim, Config.train_epoch_lr, gamma=0.5)

        self.test_tool = FSLTestTool(self.matching_test, data_root=Config.data_root,
                                     num_way=Config.num_way, num_shot=Config.num_shot,
                                     episode_size=Config.episode_size, test_episode=Config.test_episode,
                                     transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load proto net success from {}".format(Config.mn_dir))
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
        z_support = sample_z.view(Config.num_way * Config.num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, Config.num_way * Config.num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, Config.num_way * Config.num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, Config.num_way, Config.num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def train(self):
        Tools.print()
        Tools.print("Training...")

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
            Tools.print("{:6} loss:{:.3f} lr:{}".format(
                epoch, all_loss / len(self.task_train_loader), self.matching_net_scheduler.get_last_lr()))

            self.matching_net_scheduler.step()
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.val_freq == 0:
                Tools.print()
                Tools.print("Test {} {} .......".format(epoch, Config.model_name))
                self.matching_net.eval()

                val_accuracy = self.test_tool.val(episode=epoch, is_print=True, has_test=False)
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.matching_net.state_dict(), Config.mn_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


class Config(object):
    gpu_id = "0,1,2,3"
    # gpu_id = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    train_epoch = 100
    train_epoch_lr = [50, 80]
    learning_rate = 0.001
    num_workers = 16
    # train_epoch = 300
    # train_epoch_lr = [200, 250]

    num_way = 5
    num_shot = 1
    # batch_size = 256
    # batch_size = 128
    batch_size = 64
    # batch_size = 32

    val_freq = 5
    episode_size = 15
    test_episode = 600

    load_data = False

    model_name = "{}_{}_{}_{}_{}".format(gpu_id.replace(",", ""), train_epoch, batch_size, num_way, num_shot)

    # matching_net, model_name = MatchingNet(hid_dim=64, z_dim=64), "{}_{}".format(model_name, "conv4")
    matching_net, model_name = ResNet12Small(avg_pool=True, drop_rate=0.1), "{}_{}".format(model_name, "res12")

    mn_dir = Tools.new_dir("../tiered_imagenet/models_mn/fsl_modify/{}.pkl".format(model_name))
    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/tiered-imagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/tiered-imagenet'
    else:
        data_root = "F:\\data\\tiered-imagenet"

    Tools.print(model_name)
    Tools.print(data_root)
    Tools.print(mn_dir)
    pass


##############################################################################################################


"""
2020-12-20 09:37:27 load proto net success from ../tiered_imagenet/models_mn/fsl_modify/0_100_64_5_1_conv4.pkl
2020-12-20 09:37:55 Train 100 Accuracy: 0.6654444444444444
2020-12-20 09:38:22 Val   100 Accuracy: 0.5221111111111111
2020-12-20 09:38:51 Test1 100 Accuracy: 0.5603333333333333
2020-12-20 09:39:56 Test2 100 Accuracy: 0.5510444444444444
2020-12-20 09:45:18 episode=100, Test accuracy=0.5572
2020-12-20 09:45:18 episode=100, Test accuracy=0.5606444444444444
2020-12-20 09:45:18 episode=100, Test accuracy=0.5591111111111111
2020-12-20 09:45:18 episode=100, Test accuracy=0.5607333333333333
2020-12-20 09:45:18 episode=100, Test accuracy=0.5621111111111112
2020-12-20 09:45:18 episode=100, Mean Test accuracy=0.55996

2020-12-31 16:59:14 Test 100 0123_100_256_5_1_conv4 .......
2020-12-31 16:59:47 Train 100 Accuracy: 0.6671111111111111
2020-12-31 17:00:21 Val   100 Accuracy: 0.525
2020-12-31 17:00:55 Test1 100 Accuracy: 0.5589999999999999
2020-12-31 17:00:55 load proto net success from ../tiered_imagenet/models_mn/fsl_modify/0123_100_256_5_1_conv4.pkl
2020-12-31 17:01:27 Train 100 Accuracy: 0.6713333333333332
2020-12-31 17:02:00 Val   100 Accuracy: 0.5263333333333333
2020-12-31 17:02:34 Test1 100 Accuracy: 0.5511111111111111
2020-12-31 17:04:37 Test2 100 Accuracy: 0.5597333333333334
2020-12-31 17:14:53 episode=100, Test accuracy=0.5559111111111111
2020-12-31 17:14:53 episode=100, Test accuracy=0.5517555555555556
2020-12-31 17:14:53 episode=100, Test accuracy=0.5610666666666666
2020-12-31 17:14:53 episode=100, Test accuracy=0.5578
2020-12-31 17:14:53 episode=100, Test accuracy=0.5526888888888889
2020-12-31 17:14:53 episode=100, Mean Test accuracy=0.5558444444444445


2021-01-01 23:00:23 ../tiered_imagenet/models_mn/fsl_modify/0123_100_64_5_1_res12.pkl
2021-01-01 23:00:28 load proto net success from ../tiered_imagenet/models_mn/fsl_modify/0123_100_64_5_1_res12.pkl
2021-01-01 23:01:37 Train 100 Accuracy: 0.95
2021-01-01 23:02:37 Val   100 Accuracy: 0.6306666666666666
2021-01-01 23:03:36 Test1 100 Accuracy: 0.6471111111111111
2021-01-01 23:07:14 Test2 100 Accuracy: 0.6316666666666667
2021-01-01 23:25:08 episode=100, Test accuracy=0.6417555555555555
2021-01-01 23:25:08 episode=100, Test accuracy=0.6391555555555556
2021-01-01 23:25:08 episode=100, Test accuracy=0.6492666666666667
2021-01-01 23:25:08 episode=100, Test accuracy=0.6506666666666666
2021-01-01 23:25:08 episode=100, Test accuracy=0.6409555555555556
2021-01-01 23:25:08 episode=100, Mean Test accuracy=0.64436

"""


if __name__ == '__main__':
    runner = Runner()
    # runner.load_model()

    # runner.matching_net.eval()
    # runner.test_tool.val(episode=0, is_print=True)

    # runner.train()

    runner.load_model()
    runner.matching_net.eval()
    runner.test_tool.val(episode=Config.train_epoch, is_print=True)
    runner.test_tool.test(test_avg_num=5, episode=Config.train_epoch, is_print=True)
    pass
