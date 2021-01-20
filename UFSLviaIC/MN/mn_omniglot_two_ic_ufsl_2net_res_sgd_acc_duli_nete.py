import os
import sys
import time
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
from mn_tool_ic_test import ICTestTool
from mn_tool_fsl_test import FSLTestTool
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34
from torch.optim.lr_scheduler import StepLR, MultiStepLR
sys.path.append("../Common")
from UFSLTool import ResNet12Small, C4Net, RunnerTool, Normalize


##############################################################################################################


class TieredImageNetICDataset(object):

    def __init__(self, data_list, image_size=28):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
        # self.transform_train_ic = transforms.Compose([transforms.RandomRotation(30),
        #                                               transforms.Resize(image_size),
        #                                               transforms.RandomCrop(image_size, padding=4, fill=255),
        #                                               transforms.ToTensor(), normalize])
        self.transform_train_ic = transforms.Compose([transforms.RandomRotation(30, fill=255),
                                                      transforms.Resize(image_size),
                                                      transforms.RandomCrop(image_size, padding=4, fill=255),
                                                      transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = self.transform_train_ic(image)
        return image, label, idx

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

    pass


class TieredImageNetFSLDataset(object):

    def __init__(self, data_list, classes, num_way, num_shot, image_size=28):
        self.num_way, self.num_shot = num_way, num_shot
        self.true_label = [one[1] for one in data_list]
        self.data_list = [(one[0], class_one, one[2]) for one, class_one in zip(data_list, classes)]

        self.data_dict = {}
        for index, label, image_filename in self.data_list:
            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append((index, label, image_filename))
            pass
        Tools.print("class number = {}".format(len(self.data_dict.keys())))

        normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
        # self.transform_train_fsl = transforms.Compose([transforms.RandomRotation(30),
        #                                                transforms.Resize(image_size),
        #                                                transforms.RandomCrop(image_size, padding=4, fill=255),
        #                                                transforms.ToTensor(), normalize])
        self.transform_train_fsl = transforms.Compose([transforms.RandomRotation(30, fill=255),
                                                       transforms.Resize(image_size),
                                                       transforms.RandomCrop(image_size, padding=4, fill=255),
                                                       transforms.ToTensor(), normalize])
        self.transform_test = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # 当前样本
        now_label_image_tuple = self.data_list[item]
        now_index, now_label, now_image_filename = now_label_image_tuple
        now_label_k_shot_image_tuple = random.sample(self.data_dict[now_label], self.num_shot)
        is_ok_list = [self.true_label[one[0]] == self.true_label[item] for one in now_label_k_shot_image_tuple]

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
        task_data = torch.cat([torch.unsqueeze(self.read_image(one, self.transform_train_fsl),
                                               dim=0) for one in task_list])
        task_label = torch.Tensor([int(one_tuple[1] == now_label) for one_tuple in c_way_k_shot_tuple_list])
        task_index = torch.Tensor([one[0] for one in task_list]).long()
        return task_data, task_label, task_index, is_ok_list

    @staticmethod
    def read_image(one, transform=None):
        image = Image.open(one[2]).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    pass


##############################################################################################################


class EncoderC4(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = C4Net(64, 64)
        self.out_dim = 64
        pass

    def forward(self, x):
        out = self.encoder(x)
        out = torch.flatten(out, 1)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ICResNet(nn.Module):

    def __init__(self, encoder, low_dim=512):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(self.encoder.out_dim, low_dim)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.encoder(x)
        out_logits = self.fc(out)
        out_l2norm = self.l2norm(out_logits)
        return out_logits, out_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ProduceClass(object):

    def __init__(self, n_sample, out_dim, ratio=1.0):
        super().__init__()
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.class_per_num = self.n_sample // self.out_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = np.zeros(shape=(self.out_dim,), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample,), dtype=np.int)
        pass

    def init(self):
        class_per_num = self.n_sample // self.out_dim
        self.class_num += class_per_num
        for i in range(self.out_dim):
            self.classes[i * class_per_num: (i + 1) * class_per_num] = i
            pass
        np.random.shuffle(self.classes)
        pass

    def reset(self):
        self.count = 0
        self.count_2 = 0
        self.class_num *= 0
        pass

    def cal_label(self, out, indexes):
        out_data = out.data.cpu()
        top_k = out_data.topk(self.out_dim, dim=1)[1].cpu()
        indexes_cpu = indexes.cpu()

        batch_size = top_k.size(0)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int)

        for i in range(batch_size):
            for j_index, j in enumerate(top_k[i]):
                if self.class_per_num > self.class_num[j]:
                    class_labels[i] = j
                    self.class_num[j] += 1
                    self.count += 1 if self.classes[indexes_cpu[i]] != j else 0
                    self.classes[indexes_cpu[i]] = j
                    self.count_2 += 1 if j_index != 0 else 0
                    break
                pass
            pass
        pass

    def get_label(self, indexes):
        _device = indexes.device
        return torch.tensor(self.classes[indexes.cpu()]).long().to(_device)

    pass


##############################################################################################################


class RunnerIC(object):

    def __init__(self):
        self.adjust_learning_rate = Config.ic_adjust_learning_rate

        # data
        self.data_train = TieredImageNetICDataset.get_data_all(Config.data_root)
        self.tiered_imagenet_dataset = TieredImageNetICDataset(self.data_train)
        self.ic_train_loader = DataLoader(self.tiered_imagenet_dataset, Config.ic_batch_size,
                                          shuffle=True, num_workers=Config.num_workers)
        self.ic_train_loader_eval = DataLoader(self.tiered_imagenet_dataset, Config.ic_batch_size,
                                               shuffle=False, num_workers=Config.num_workers)

        # IC
        self.produce_class = ProduceClass(len(self.data_train), Config.ic_out_dim, Config.ic_ratio)
        self.produce_class.init()

        # model
        self.ic_model = RunnerTool.to_cuda(ICResNet(low_dim=Config.ic_out_dim, encoder=Config.ic_net))
        RunnerTool.to_cuda(self.ic_model.apply(RunnerTool.weights_init))
        self.ic_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())

        self.ic_model_optim = torch.optim.SGD(
            self.ic_model.parameters(), lr=Config.ic_learning_rate, momentum=0.9, weight_decay=5e-4)

        # Eval
        self.test_tool_ic = ICTestTool(feature_encoder=None, ic_model=self.ic_model,
                                       data_root=Config.data_root, batch_size=Config.ic_batch_size,
                                       num_workers=Config.num_workers, ic_out_dim=Config.ic_out_dim,
                                       transform=self.ic_train_loader.dataset.transform_test, k=Config.ic_knn)
        pass

    def load_model(self, ic_dir_checkpoint=None):
        ic_dir = ic_dir_checkpoint if ic_dir_checkpoint else Config.ic_dir
        if os.path.exists(ic_dir):
            checkpoint = torch.load(ic_dir)
            if ic_dir_checkpoint:
                if not list(self.ic_model.state_dict().keys())[0] == list(checkpoint.keys())[0]:
                    checkpoint = {"module.{}".format(key): checkpoint[key] for key in checkpoint}
            self.ic_model.load_state_dict(checkpoint)
            Tools.print("load ic model success from {}".format(ic_dir))
        pass

    def train(self):
        Tools.print()
        Tools.print("Training...")
        best_accuracy = 0.0

        # Init Update
        try:
            self.ic_model.eval()
            Tools.print("Init label {} .......")
            self.produce_class.reset()
            with torch.no_grad():
                for image, label, idx in tqdm(self.ic_train_loader):
                    image, idx = RunnerTool.to_cuda(image), RunnerTool.to_cuda(idx)
                    ic_out_logits, ic_out_l2norm = self.ic_model(image)
                    self.produce_class.cal_label(ic_out_l2norm, idx)
                    pass
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2))
        finally:
            pass

        for epoch in range(1, 1 + Config.ic_train_epoch):
            self.ic_model.train()

            Tools.print()
            ic_lr = self.adjust_learning_rate(self.ic_model_optim, epoch, Config.ic_first_epoch,
                                              Config.ic_t_epoch, Config.ic_learning_rate)
            Tools.print('Epoch: [{}] ic_lr={}'.format(epoch, ic_lr))

            all_loss = 0.0
            self.produce_class.reset()
            for image, label, idx in tqdm(self.ic_train_loader):
                image, label, idx = RunnerTool.to_cuda(image), RunnerTool.to_cuda(label), RunnerTool.to_cuda(idx)

                ###########################################################################
                # 1 calculate features
                ic_out_logits, ic_out_l2norm = self.ic_model(image)

                # 2 calculate labels
                ic_targets = self.produce_class.get_label(idx)
                self.produce_class.cal_label(ic_out_l2norm, idx)

                # 3 loss
                loss = self.ic_loss(ic_out_logits, ic_targets)
                all_loss += loss.item()

                # 4 backward
                self.ic_model.zero_grad()
                loss.backward()
                self.ic_model_optim.step()
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.4f}".format(epoch, all_loss / len(self.ic_train_loader)))
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.ic_val_freq == 0:
                self.ic_model.eval()

                val_accuracy = self.test_tool_ic.val(epoch=epoch)
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.ic_model.state_dict(), Config.ic_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    def eval(self):
        self.ic_model.eval()
        with torch.no_grad():
            ic_classes = np.zeros(shape=(len(self.tiered_imagenet_dataset),), dtype=np.int)
            for image, label, idx in tqdm(self.ic_train_loader_eval):
                ic_out_logits, ic_out_l2norm = self.ic_model(RunnerTool.to_cuda(image))
                ic_classes[idx] = np.argmax(ic_out_l2norm.cpu().numpy(), axis=-1)
                pass
        return self.data_train, ic_classes

    pass


class RunnerFSL(object):

    def __init__(self, data_train, classes):
        # all data
        self.data_train = data_train
        self.task_train = TieredImageNetFSLDataset(self.data_train, classes, Config.fsl_num_way, Config.fsl_num_shot)
        self.task_train_loader = DataLoader(self.task_train, Config.fsl_batch_size, True, num_workers=Config.num_workers)

        # model
        self.matching_net = RunnerTool.to_cuda(Config.fsl_matching_net)
        self.matching_net = RunnerTool.to_cuda(nn.DataParallel(self.matching_net))
        cudnn.benchmark = True
        RunnerTool.to_cuda(self.matching_net.apply(RunnerTool.weights_init))
        self.norm = Normalize(2)

        # optim
        self.matching_net_optim = torch.optim.Adam(self.matching_net.parameters(), lr=Config.fsl_learning_rate)
        self.matching_net_scheduler = MultiStepLR(self.matching_net_optim, Config.fsl_lr_schedule, gamma=1/3)

        # loss
        self.fsl_loss = RunnerTool.to_cuda(nn.MSELoss())

        # Eval
        self.test_tool_fsl = FSLTestTool(self.matching_test, data_root=Config.data_root,
                                         num_way=Config.fsl_num_way, num_shot=Config.fsl_num_shot,
                                         episode_size=Config.fsl_episode_size, test_episode=Config.fsl_test_episode,
                                         transform=self.task_train.transform_test)
        pass

    def load_model(self):
        if os.path.exists(Config.mn_dir):
            self.matching_net.load_state_dict(torch.load(Config.mn_dir))
            Tools.print("load matching net success from {}".format(Config.mn_dir))
        pass

    def matching(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out = self.matching_net(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        # 特征
        z_support, z_query = z.split(Config.fsl_num_shot * Config.fsl_num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, Config.fsl_num_way * Config.fsl_num_shot, z_dim)
        z_query_expand = z_query.expand(z_batch_size, Config.fsl_num_way * Config.fsl_num_shot, z_dim)

        # 相似性
        z_support = self.norm(z_support)
        similarities = torch.sum(z_support * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(z_batch_size, Config.fsl_num_way, Config.fsl_num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def matching_test(self, samples, batches):
        batch_num, _, _, _ = batches.shape

        sample_z = self.matching_net(samples)  # 5x64*5*5
        batch_z = self.matching_net(batches)  # 75x64*5*5
        z_support = sample_z.view(Config.fsl_num_way * Config.fsl_num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, Config.fsl_num_way * Config.fsl_num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, Config.fsl_num_way * Config.fsl_num_shot, z_dim)

        # 相似性
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, Config.fsl_num_way, Config.fsl_num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def train(self):
        Tools.print()
        Tools.print("Training...")
        best_accuracy = 0.0

        for epoch in range(1, 1 + Config.fsl_train_epoch):
            self.matching_net.train()

            Tools.print()
            all_loss, is_ok_total, is_ok_acc = 0.0, 0, 0
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                relations = self.matching(task_data)

                # 3 loss
                loss = self.fsl_loss(relations, task_labels)
                all_loss += loss.item()

                # 4 backward
                self.matching_net.zero_grad()
                loss.backward()
                self.matching_net_optim.step()

                # is ok
                is_ok_acc += torch.sum(torch.cat(task_ok))
                is_ok_total += torch.prod(torch.tensor(torch.cat(task_ok).shape))
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} ok:{:.3f}({}/{}) lr:{}".format(
                epoch, all_loss / len(self.task_train_loader), int(is_ok_acc) / int(is_ok_total),
                is_ok_acc, is_ok_total, self.matching_net_scheduler.get_last_lr()))
            self.matching_net_scheduler.step()
            ###########################################################################

            ###########################################################################
            # Val
            if epoch % Config.fsl_val_freq == 0:
                self.matching_net.eval()

                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True, has_test=False)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.matching_net.state_dict(), Config.mn_dir)
                    Tools.print("Save networks for epoch: {}".format(epoch))
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


##############################################################################################################


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 32

    #######################################################################################
    ic_ratio = 1
    ic_knn = 100
    # ic_out_dim = 2048
    ic_out_dim = 1024
    ic_val_freq = 10

    ic_net, ic_net_name = EncoderC4(), "ICConv4"

    ic_learning_rate = 0.01
    ic_train_epoch = 1200
    ic_first_epoch, ic_t_epoch = 400, 200
    ic_batch_size = 64 * 4

    ic_adjust_learning_rate = RunnerTool.adjust_learning_rate1
    #######################################################################################

    ###############################################################################################
    fsl_num_way = 5
    fsl_num_shot = 1

    fsl_episode_size = 15
    fsl_test_episode = 600

    fsl_matching_net, fsl_net_name, fsl_batch_size = C4Net(hid_dim=64, z_dim=64), "conv4", 96

    fsl_learning_rate = 0.01
    fsl_batch_size = fsl_batch_size
    fsl_val_freq = 10
    fsl_train_epoch = 300
    fsl_lr_schedule = [150, 250]
    ###############################################################################################

    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        gpu_id, ic_net_name, ic_train_epoch, ic_batch_size, ic_out_dim,
        fsl_net_name, fsl_train_epoch, fsl_num_way, fsl_num_shot, fsl_batch_size)

    dataset_name = "omniglot_single"
    # dataset_name = "omniglot_rot"
    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/UFSL/{}'.format(dataset_name)
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/{}'.format(dataset_name)
        if not os.path.isdir(data_root):
            data_root = '/home/ubuntu/Dataset/Partition1/ALISURE/Data/UFSL/{}'.format(dataset_name)
    else:
        data_root = "F:\\data\\{}".format(dataset_name)
    Tools.print(data_root)

    _root_path = "../omniglot/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete"
    mn_dir = Tools.new_dir("{}/{}_mn.pkl".format(_root_path, model_name))
    ic_dir = Tools.new_dir("{}/{}_ic.pkl".format(_root_path, model_name))
    # ic_dir_checkpoint = None
    # ic_dir_checkpoint = "../omniglot/ic/2_28_ICConv4_64_2048_1_1500_500_200_ic.pkl"
    # ic_dir_checkpoint = "../omniglot/ic/3_28_ICConv4_64_2048_1_1600_1000_300_ic.pkl"
    ic_dir_checkpoint = "../omniglot/ic/2_28_ICConv4_64_1024_1_1600_1000_300_ic.pkl"

    Tools.print(model_name)
    pass


"""
transforms.RandomRotation(30)
../omniglot/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/1_ICConv4_1200_256_2048_conv4_300_5_1_96_77_mn.pkl
2021-01-19 23:20:15 load matching net success from ../omniglot/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/1_ICConv4_1200_256_2048_conv4_300_5_1_96_77_mn.pkl
2021-01-19 23:20:24 Train 300 Accuracy: 0.9704444444444444
2021-01-19 23:20:34 Val   300 Accuracy: 0.9584444444444448
2021-01-19 23:20:43 Test1 300 Accuracy: 0.9524444444444445
2021-01-19 23:21:26 Test2 300 Accuracy: 0.9515555555555557
2021-01-19 23:24:23 episode=300, Test accuracy=0.9465111111111113
2021-01-19 23:24:23 episode=300, Test accuracy=0.9533555555555555
2021-01-19 23:24:23 episode=300, Test accuracy=0.9494888888888888
2021-01-19 23:24:23 episode=300, Test accuracy=0.9498666666666667
2021-01-19 23:24:23 episode=300, Test accuracy=0.9491111111111111
2021-01-19 23:24:23 episode=300, Mean Test accuracy=0.9496666666666667


transforms.RandomRotation(30, fill=255)
"../omniglot/ic/3_28_ICConv4_64_2048_1_1600_1000_300_ic.pkl"
../omniglot/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/3_ICConv4_1200_256_2048_conv4_300_5_1_96_mn.pkl
2021-01-20 10:40:58 load matching net success from ../omniglot/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/3_ICConv4_1200_256_2048_conv4_300_5_1_96_mn.pkl
2021-01-20 10:41:10 Train 300 Accuracy: 0.9767777777777779
2021-01-20 10:41:23 Val   300 Accuracy: 0.962777777777778
2021-01-20 10:41:37 Test1 300 Accuracy: 0.9573333333333333
2021-01-20 10:42:36 Test2 300 Accuracy: 0.9575555555555555
2021-01-20 10:46:55 episode=300, Test accuracy=0.9573333333333333
2021-01-20 10:46:55 episode=300, Test accuracy=0.9587333333333333
2021-01-20 10:46:55 episode=300, Test accuracy=0.9565111111111111
2021-01-20 10:46:55 episode=300, Test accuracy=0.9571999999999999
2021-01-20 10:46:55 episode=300, Test accuracy=0.9572888888888889
2021-01-20 10:46:55 episode=300, Mean Test accuracy=0.9574133333333332


transforms.RandomRotation(30, fill=255)
"../omniglot/ic/2_28_ICConv4_64_1024_1_1600_1000_300_ic.pkl"
../omniglot/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/2_ICConv4_1200_256_1024_conv4_300_5_1_96_mn.pkl
2021-01-20 10:45:33 load matching net success from ../omniglot/models_mn/two_ic_ufsl_2net_res_sgd_acc_duli_nete/2_ICConv4_1200_256_1024_conv4_300_5_1_96_mn.pkl
2021-01-20 10:45:48 Train 300 Accuracy: 0.9857777777777779
2021-01-20 10:46:02 Val   300 Accuracy: 0.9744444444444446
2021-01-20 10:46:16 Test1 300 Accuracy: 0.9696666666666668
2021-01-20 10:47:14 Test2 300 Accuracy: 0.970311111111111
2021-01-20 10:50:56 episode=300, Test accuracy=0.9695777777777778
2021-01-20 10:50:56 episode=300, Test accuracy=0.9734888888888887
2021-01-20 10:50:56 episode=300, Test accuracy=0.9699333333333332
2021-01-20 10:50:56 episode=300, Test accuracy=0.9736222222222224
2021-01-20 10:50:56 episode=300, Test accuracy=0.9717333333333332
2021-01-20 10:50:56 episode=300, Mean Test accuracy=0.9716711111111109


"""


if __name__ == '__main__':
    runner_ic = RunnerIC()
    # runner_ic.train()
    runner_ic.load_model(ic_dir_checkpoint=Config.ic_dir_checkpoint)
    data_train, classes = runner_ic.eval()

    runner_fsl = RunnerFSL(data_train=data_train, classes=classes)
    runner_fsl.train()

    runner_ic.load_model()
    runner_fsl.load_model()
    runner_ic.ic_model.eval()
    runner_fsl.matching_net.eval()
    # runner_ic.test_tool_ic.val(epoch=Config.ic_train_epoch, is_print=True)
    runner_fsl.test_tool_fsl.val(episode=Config.fsl_train_epoch, is_print=True)
    runner_fsl.test_tool_fsl.test(test_avg_num=5, episode=Config.fsl_train_epoch, is_print=True)
    pass
