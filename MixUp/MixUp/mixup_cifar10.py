import os
import math
import torch
import platform
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from alisuretool.Tools import Tools
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        pass

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        pass

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        pass

    pass


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * planes))
            pass
        pass

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    pass


class MixUpResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim=512):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, low_dim, bias=False)
        self.l2norm = Normalize(2)
        pass

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        convB1 = F.relu(self.bn1(self.conv1(x)))
        convB2 = self.layer1(convB1)
        convB3 = self.layer2(convB2)
        convB4 = self.layer3(convB3)
        convB5 = self.layer4(convB4)
        avgPool = F.avg_pool2d(convB5, 4)
        avgPool = avgPool.view(avgPool.size(0), -1)

        out_logits = self.linear(avgPool)
        out_l2norm = self.l2norm(out_logits)
        out_softmax = F.softmax(out_logits, dim=-1)
        return out_logits, out_l2norm, out_softmax

    pass


class CIFAR10MixUp(datasets.CIFAR10):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not (hasattr(self, "data") and hasattr(self, "targets")):
            if self.train:
                self.data, self.target = self.train_data, self.train_labels
            else:
                self.data, self.target = self.test_data, self.test_labels
            pass
        pass

    def __getitem__(self, index):
        img1, target1, index1 = self._getitem(index)
        index2 = np.random.randint(len(self.data))
        img2, target2, index2 = self._getitem(index2)
        img = torch.cat([torch.unsqueeze(img1, 0), torch.unsqueeze(img2, 0)], dim=0)
        target = [target1, target2]
        index = [index1, index2]
        return img, target, index

    def _getitem(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    pass


class CIFAR10Instance(datasets.CIFAR10):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not (hasattr(self, "data") and hasattr(self, "targets")):
            if self.train:
                self.data, self.target = self.train_data, self.train_labels
            else:
                self.data, self.target = self.test_data, self.test_labels
            pass
        pass

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    @staticmethod
    def data(data_root, batch_size=128):
        Tools.print('==> Preparing data..')

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = CIFAR10MixUp(root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        test_train_set = CIFAR10Instance(root=data_root, train=True, download=True, transform=transform_test)
        test_train_loader = torch.utils.data.DataLoader(test_train_set, batch_size=batch_size, shuffle=False, num_workers=2)

        test_set = CIFAR10Instance(root=data_root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_set, train_loader, test_train_set, test_train_loader, test_set, test_loader

    pass


class KNN(object):

    @staticmethod
    def knn(epoch, net, low_dim, train_loader, test_loader, k, t, loader_n=1):
        net.eval()
        n_sample = train_loader.dataset.__len__()
        out_memory = Config.cuda(torch.rand(n_sample, low_dim).t())

        targets = train_loader.dataset.train_labels if hasattr(train_loader.dataset, "train_labels") else(
            train_loader.dataset.targets if hasattr(train_loader.dataset, "targets") else train_loader.dataset.labels)
        train_labels = Config.cuda(torch.LongTensor(targets))
        max_c = train_labels.max() + 1

        transform_bak = train_loader.dataset.transform
        train_loader.dataset.transform = test_loader.dataset.transform
        temp_loader = torch.utils.data.DataLoader(train_loader.dataset, 100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, _, indexes) in enumerate(temp_loader):
            inputs = Config.cuda(inputs)
            out_logits, out_l2norm, out_softmax = net(inputs)
            batch_size = inputs.size(0)
            out_memory[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = out_l2norm.data.t()
            pass
        train_loader.dataset.transform = transform_bak

        def _cal(inputs, dist, train_labels, retrieval_one_hot, top1, top5):
            # ---------------------------------------------------------------------------------- #
            batch_size = inputs.size(0)
            yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batch_size * k, max_c).zero_()
            retrieval_one_hot = retrieval_one_hot.scatter_(1, retrieval.view(-1, 1),
                                                           1).view(batch_size, -1, max_c)
            yd_transform = yd.clone().div_(t).exp_().view(batch_size, -1, 1)
            probs = torch.sum(torch.mul(retrieval_one_hot, yd_transform), 1)
            _, predictions = probs.sort(1, True)
            # ---------------------------------------------------------------------------------- #

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1, 1))

            top1 += correct.narrow(1, 0, 1).sum().item()
            top5 += correct.narrow(1, 0, 5).sum().item()
            return top1, top5, retrieval_one_hot

        all_acc = []
        with torch.no_grad():
            now_loader = [test_loader] if loader_n == 1 else [test_loader, train_loader]
            for loader in now_loader:
                total, top1, top5 = 0, 0., 0.
                retrieval_one_hot = Config.cuda(torch.zeros(k, max_c))  # [200, 10]
                for batch_idx, (inputs, targets, indexes) in enumerate(loader):
                    targets, inputs = Config.cuda(targets), Config.cuda(inputs)
                    total += targets.size(0)

                    out_logits, out_l2norm, out_softmax = net(inputs)
                    dist = torch.mm(out_l2norm, out_memory)
                    top1, top5, retrieval_one_hot = _cal(inputs, dist, train_labels, retrieval_one_hot, top1, top5)
                    pass

                Tools.print("Test 1 {} Top1={:.2f} Top5={:.2f}".format(
                    epoch, top1 * 100. / total, top5 * 100. / total))
                all_acc.append(top1 / total)
                pass
            pass

        return all_acc[0]

    pass


class ProduceClass(object):

    def __init__(self, n_sample, low_dim, ratio=1.0):
        self.low_dim = low_dim
        self.n_sample = n_sample
        self.class_per_num = self.n_sample // self.low_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = np.zeros(shape=(self.low_dim, ), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample, ), dtype=np.int)
        pass

    def init(self):
        class_per_num = self.n_sample // self.low_dim
        self.class_num += class_per_num
        for i in range(self.low_dim):
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
        top_k = out.data.topk(self.low_dim, dim=1)[1].cpu()
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
        return Config.cuda(torch.tensor(self.classes[indexes.cpu()]).long())

    pass


class MixUpRunner(object):

    def __init__(self):
        self.best_acc = 0
        self.train_set, self.train_loader, self.test_train_set, self.test_train_loader, self.test_set, \
        self.test_loader = CIFAR10Instance.data(Config.data_root(), batch_size=Config.batch_size)

        self.train_num = self.train_set.__len__()

        self.net = Config.cuda(MixUpResNet(BasicBlock, [2, 2, 2, 2], Config.low_dim))

        self._load_model(self.net)

        self.produce_class = ProduceClass(n_sample=self.train_num, low_dim=Config.low_dim, ratio=Config.ratio)
        self.produce_class.init()

        self.criterion = Config.cuda(nn.CrossEntropyLoss())
        self.optimizer = optim.SGD(self.net.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        pass

    def _adjust_learning_rate(self, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=Config.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = Config.t_epoch
        first_epoch = Config.first_epoch
        init_learning_rate = Config.learning_rate
        if epoch < first_epoch + t_epoch * 0:  # 0-200
            learning_rate = init_learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 200-300
            learning_rate = init_learning_rate / 2
        elif epoch < first_epoch + t_epoch * 2:  # 300-400
            learning_rate = init_learning_rate / 4
        elif epoch < first_epoch + t_epoch * 3:  # 400-500
            learning_rate = _get_lr(init_learning_rate / 2.0, epoch - first_epoch - t_epoch * 2)
        elif epoch < first_epoch + t_epoch * 4:  # 500-600
            learning_rate = _get_lr(init_learning_rate / 2.0, epoch - first_epoch - t_epoch * 3)
        elif epoch < first_epoch + t_epoch * 5:  # 600-700
            learning_rate = _get_lr(init_learning_rate / 4.0, epoch - first_epoch - t_epoch * 4)
        elif epoch < first_epoch + t_epoch * 6:  # 700-800
            learning_rate = _get_lr(init_learning_rate / 4.0, epoch - first_epoch - t_epoch * 5)
        elif epoch < first_epoch + t_epoch * 7:  # 800-900
            learning_rate = _get_lr(init_learning_rate / 8.0, epoch - first_epoch - t_epoch * 6)
        elif epoch < first_epoch + t_epoch * 8:  # 900-1000
            learning_rate = _get_lr(init_learning_rate / 8.0, epoch - first_epoch - t_epoch * 7)
        elif epoch < first_epoch + t_epoch * 9:  # 1000-1100
            learning_rate = _get_lr(init_learning_rate / 16., epoch - first_epoch - t_epoch * 8)
        elif epoch < first_epoch + t_epoch * 10:  # 1100-1200
            learning_rate = _get_lr(init_learning_rate / 16., epoch - first_epoch - t_epoch * 9)
        elif epoch < first_epoch + t_epoch * 11:  # 1200-1300
            learning_rate = _get_lr(init_learning_rate / 32., epoch - first_epoch - t_epoch * 10)
        elif epoch < first_epoch + t_epoch * 12:  # 1300-1400
            learning_rate = _get_lr(init_learning_rate / 32., epoch - first_epoch - t_epoch * 11)
        elif epoch < first_epoch + t_epoch * 13:  # 1400-1500
            learning_rate = _get_lr(init_learning_rate / 64., epoch - first_epoch - t_epoch * 12)
        else:  # 1500-1600
            learning_rate = _get_lr(init_learning_rate / 64., epoch - first_epoch - t_epoch * 13)
            pass

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass

        return learning_rate

    def _load_model(self, net):
        # Load PreTrain
        if Config.pre_train:
            Tools.print('==> Pre train from checkpoint {} ..'.format(Config.pre_train))
            checkpoint = torch.load(Config.pre_train)
            net.load_state_dict(checkpoint['net'], strict=False)
            self.best_acc = checkpoint['acc']
            best_epoch = checkpoint['epoch']
            Tools.print("{} {}".format(self.best_acc, best_epoch))
            pass

        # Load checkpoint.
        if Config.resume:
            Tools.print('==> Resuming from checkpoint {} ..'.format(Config.checkpoint_path))
            checkpoint = torch.load(Config.checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            best_epoch = checkpoint['epoch']
            Tools.print("{} {}".format(self.best_acc, best_epoch))
            pass
        pass

    def _train_one_epoch(self, epoch):
        # Train
        try:
            self.net.train()
            _learning_rate_ = self._adjust_learning_rate(epoch)
            Tools.print('Epoch: [{}] lr={} name={}'.format(epoch, _learning_rate_, Config.name))

            self.produce_class.reset()
            avg_loss, avg_loss_1, avg_loss_2 = AverageMeter(), AverageMeter(), AverageMeter()
            for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                batch_size, _, c, w, h = inputs.shape
                a = np.random.beta(1, 1, [batch_size, 1])
                b = np.hstack([a, 1 - a])
                ab = np.tile(b[..., None, None, None], [c, w, h])

                x1_add_x2 = torch.mean(inputs * ab, dim=1, keepdim=True).float()
                now_inputs = torch.cat([inputs, x1_add_x2], dim=1).view(-1, c, w, h)
                now_inputs = Config.cuda(now_inputs)

                self.optimizer.zero_grad()

                out_logits, out_l2norm, out_softmax = self.net(now_inputs)
                out_dim = out_softmax.shape[-1]
                now_out_logits = out_logits.view(batch_size, -1, out_dim)
                now_out_l2norm = out_l2norm.view(batch_size, -1, out_dim)
                now_out_softmax = out_softmax.view(batch_size, -1, out_dim)

                softmax_w = np.hstack([a, 1 - a, -np.ones_like(a)])
                softmax_w = torch.tensor(np.tile(softmax_w[..., None], [out_dim,])).float()
                now_softmax_w = Config.cuda(softmax_w)
                out_error = torch.sum(now_out_softmax * now_softmax_w, dim=1)
                loss1 = torch.mean(torch.sum(torch.pow(out_error, 2), dim=1))
                loss1 *=  Config.mix_ratio

                targets = self.produce_class.get_label(indexes[0])
                targets2 = self.produce_class.get_label(indexes[1])
                self.produce_class.cal_label(now_out_l2norm[:, 0, :], indexes[0])
                loss2 = self.criterion(now_out_logits[:, 0, :], targets)

                if Config.has_ic_2:
                    loss2_2 = self.criterion(now_out_logits[:, 1, :], targets2)
                    loss2 = loss2 + loss2_2
                    pass

                loss = (loss1 + loss2) if Config.has_mix else loss2
                loss.backward()
                self.optimizer.step()

                avg_loss.update(loss.item(), inputs.size(0))
                avg_loss_1.update(loss1.item(), inputs.size(0))
                avg_loss_2.update(loss2.item(), inputs.size(0))

                # if batch_idx % 100 == 0:
                #     Tools.print('Train: [{}] {}/{} Loss: {:.4f}/{:.4f} '
                #                 'Loss 1: {:.4f}/{:.4f} Loss 2: {:.4f}/{:.4f}'.format(
                #         epoch, batch_idx, len(self.train_loader), avg_loss.avg, avg_loss.val,
                #         avg_loss_1.avg, avg_loss_1.val, avg_loss_2.avg, avg_loss_2.val))
                pass

            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2))
            Tools.print('Train: [{}] {} Loss: {:.4f}/{:.4f} '
                        'Loss 1: {:.4f}/{:.4f} Loss 2: {:.4f}/{:.4f}'.format(
                        epoch, len(self.train_loader), avg_loss.avg, avg_loss.val,
                        avg_loss_1.avg, avg_loss_1.val, avg_loss_2.avg, avg_loss_2.val))
        finally:
            pass

        # Test
        try:
            Tools.print("Test:  [{}] .......".format(epoch))
            _acc = self.test(epoch=epoch)
            if _acc > self.best_acc:
                Tools.print('Saving..')
                state = {'net': self.net.state_dict(), 'acc': _acc, 'epoch': epoch}
                torch.save(state, Config.checkpoint_path)
                self.best_acc = _acc
                pass
            Tools.print('Test:  [{}] best accuracy: {:.2f}'.format(epoch, self.best_acc * 100))
        finally:
            pass

        pass

    def train(self, start_epoch):
        for epoch in range(start_epoch, Config.max_epoch):
            Tools.print()
            self._train_one_epoch(epoch)
            pass
        pass

    def test(self, epoch=0, t=0.1, loader_n=1):
        _acc = KNN.knn(epoch, self.net, Config.low_dim,
                       self.test_train_loader, self.test_loader, 200, t, loader_n=loader_n)
        return _acc

    pass


class Config(object):
    gpu_id = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    start_epoch = 0
    max_epoch = 1600
    learning_rate = 0.01
    first_epoch, t_epoch = 200, 100
    low_dim = 512
    ratio = 2

    mix_ratio = 100.0

    if gpu_id == "0":
        has_mix = False
        has_ic_2 = True
    elif gpu_id == "1":
        has_mix = True
        has_ic_2 = True
    elif gpu_id == "2":
        has_mix = False
        has_ic_2 = False
    elif gpu_id == "3":
        has_mix = True
        has_ic_2 = False
    else:
        raise Exception(".....")

    batch_size = 64
    resume = False
    pre_train = None

    name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(low_dim, max_epoch, batch_size, first_epoch,
                                               t_epoch, learning_rate, has_mix, has_ic_2, mix_ratio)
    checkpoint_path = Tools.new_dir("../models_mix_up/{}/ckpt.t7".format(name))
    Tools.print(name)
    Tools.print(checkpoint_path)

    @staticmethod
    def data_root():
        if "Linux" in platform.platform():
            data_root = "/mnt/4T/Data/data/CIFAR"
            if not os.path.isdir(data_root):
                data_root = '/media/ubuntu/4T/ALISURE/Data/cifar'
        else:
            data_root  ="F:\\data\\cifar10"
        return data_root

    @staticmethod
    def cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    pass


"""
512_1600_64_200_100_0.01_True_True_1.0
2020-11-05 03:57:59 Train: [1599] 7909/2085
2020-11-05 03:57:59 Train: [1599] 782 Loss: 1.4198/2.0194 Loss 1: 0.1600/0.1309 Loss 2: 1.2597/1.8885
2020-11-05 03:58:14 Test 1 1599 Top1=84.27 Top5=99.37
2020-11-05 03:58:14 Test:  [1599] best accuracy: 84.60
2020-11-05 03:58:29 Test 1 0 Top1=84.27 Top5=99.37
2020-11-05 03:58:40 Test 1 0 Top1=84.33 Top5=99.79
2020-11-05 03:58:40 final accuracy: 84.27

512_1600_64_200_100_0.01_False_True_1.0
2020-11-05 04:19:22 Train: [1599] 7953/2168
2020-11-05 04:19:22 Train: [1599] 782 Loss: 1.2596/2.1921 Loss 1: 0.4673/0.4318 Loss 2: 1.2596/2.1921
2020-11-05 04:19:37 Test 1 1599 Top1=83.65 Top5=99.34
2020-11-05 04:19:37 Test:  [1599] best accuracy: 84.18
2020-11-05 04:19:52 Test 1 0 Top1=83.65 Top5=99.34
2020-11-05 04:20:02 Test 1 0 Top1=83.63 Top5=99.71
2020-11-05 04:20:02 final accuracy: 83.65

512_1600_64_200_100_0.01_True_False_1.0
2020-11-05 04:04:33 Train: [1599] 12782/2218
2020-11-05 04:04:33 Train: [1599] 782 Loss: 1.1363/1.8521 Loss 1: 0.1157/0.1250 Loss 2: 1.0207/1.7271
2020-11-05 04:04:33 Test:  [1599] .......
2020-11-05 04:04:48 Test 1 1599 Top1=83.87 Top5=99.39
2020-11-05 04:04:48 Test:  [1599] best accuracy: 83.93
2020-11-05 04:05:04 Test 1 0 Top1=83.87 Top5=99.39
2020-11-05 04:05:13 Test 1 0 Top1=84.13 Top5=99.72
2020-11-05 04:05:13 final accuracy: 83.87
"""


if __name__ == '__main__':
    runner = MixUpRunner()
    Tools.print()
    # acc = runner.test()
    # Tools.print('Random accuracy: {:.2f}'.format(acc * 100))
    runner.train(start_epoch=Config.start_epoch)
    Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('final accuracy: {:.2f}'.format(acc * 100))
    pass
