import os
import torch
import random
import visdom
import platform
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))
            pass
        pass

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    pass


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        pass

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, [out1, out2, out3, out4]

    pass


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)
        pass

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out

    pass


class SubsetSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    pass


class Runner(object):

    def __init__(self, device):
        self.device = device
        pass

    @staticmethod
    def set_seed():
        random.seed("ALISURE")
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        pass

    @staticmethod
    def get_data(data_root):
        train_transform = T.Compose([
            T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
        ])

        cifar10_train = CIFAR10(data_root, train=True, download=True, transform=train_transform)
        cifar10_unlabeled = CIFAR10(data_root, train=True, download=True, transform=test_transform)
        cifar10_test = CIFAR10(data_root, train=False, download=True, transform=test_transform)
        return cifar10_train, cifar10_unlabeled, cifar10_test

    @staticmethod
    def loss_pred_loss(input, target, margin=1.0, reduction='mean'):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[:len(input) // 2]  # [l_1-l_2B,l_2-l_2B-1,...,l_B-l_B+1], where batch_size=2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            raise NotImplementedError()
        return loss

    def get_uncertainty(self, models, unlabeled_loader):
        models['backbone'].eval()
        models['module'].eval()
        uncertainty = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for (inputs, labels) in unlabeled_loader:
                inputs = inputs.to(self.device)

                scores, features = models['backbone'](inputs)
                # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))

                uncertainty = torch.cat((uncertainty, pred_loss), 0)
            pass

        return uncertainty.cpu()

    def train(self, models, criterion, optimizers, schedulers, data_loaders, num_epochs, epoch_loss):
        Tools.print('>> Train a Model.')

        for epoch in range(num_epochs):
            Tools.print("Epoch {}/{}".format(epoch, num_epochs))

            models['backbone'].train()
            models['module'].train()
            for step, data in tqdm(enumerate(data_loaders['train']), leave=False, total=len(data_loaders['train'])):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                optimizers['backbone'].zero_grad()
                optimizers['module'].zero_grad()

                scores, features = models['backbone'](inputs)
                target_loss = criterion(scores, labels)

                if epoch > epoch_loss:
                    # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                    features[0], features[1] = features[0].detach(), features[1].detach()
                    features[2], features[3] = features[2].detach(), features[3].detach()
                    pass

                pred_loss = models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))

                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                m_module_loss = self.loss_pred_loss(pred_loss, target_loss, margin=Config.margin)
                loss = m_backbone_loss + Config.weight * m_module_loss

                loss.backward()
                optimizers['backbone'].step()
                optimizers['module'].step()

                # Visualize
                if Config.vis_vis is not None and Config.vis_plot_data is not None and \
                                        (Config.vis_step_count + step) % 100 == 0:
                    Config.vis_plot_data['X'].append(Config.vis_step_count + step)
                    Config.vis_plot_data['Y'].append([m_backbone_loss.item(), m_module_loss.item(), loss.item()])
                    Config.vis_vis.line(
                        X=np.stack([np.array(Config.vis_plot_data['X'])] * len(Config.vis_plot_data['legend']), 1),
                        Y=np.array(Config.vis_plot_data['Y']),
                        opts={
                            'title': 'Loss over Time',
                            'legend': Config.vis_plot_data['legend'],
                            'xlabel': 'Iterations',
                            'ylabel': 'Loss',
                            'width': 1200,
                            'height': 390,
                        }, win=1)
                    pass

                pass

            schedulers['backbone'].step()
            schedulers['module'].step()
            pass

        Tools.print('>> Finished.')
        pass

    def test(self, models, data_loaders, mode='val'):
        assert mode == 'val' or mode == 'test'
        models['backbone'].eval()
        models['module'].eval()

        total = 0
        correct = 0
        with torch.no_grad():
            for (inputs, labels) in data_loaders[mode]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                scores, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
            pass

        return 100 * correct / total

    def main(self, trial=0):
        cifar10_train, cifar10_unlabeled, cifar10_test = self.get_data(data_root=Config.data_root)

        indices = list(range(len(cifar10_train)))
        random.shuffle(indices)
        labeled_set = indices[:Config.addendum]
        unlabeled_set = indices[Config.addendum:]

        # Data
        train_loader = DataLoader(cifar10_train, batch_size=Config.batch_size,
                                  sampler=SubsetRandomSampler(labeled_set), pin_memory=True)
        test_loader = DataLoader(cifar10_test, batch_size=Config.batch_size)
        data_loaders = {'train': train_loader, 'test': test_loader}

        # Model
        resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(self.device)
        loss_module = LossNet().to(self.device)
        models = {'backbone': resnet18, 'module': loss_module}
        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(Config.cycles):
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=Config.lr, momentum=0.9, weight_decay=5e-4)
            optim_module = optim.SGD(models['module'].parameters(), lr=Config.lr, momentum=0.9, weight_decay=5e-4)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=Config.milestones)
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=Config.milestones)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            self.train(models, criterion, optimizers, schedulers, data_loaders, Config.epoch, Config.epoch_stop)
            acc = self.test(models, data_loaders, mode='test')
            Tools.print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(
                trial + 1, Config.trials, cycle + 1, Config.cycles, len(labeled_set), acc))

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:Config.subset]

            # Create unlabeled dataloader for the unlabeled subset and Measure uncertainty of each data points
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=Config.batch_size,
                                          sampler=SubsetSequentialSampler(subset), pin_memory=True)
            uncertainty = self.get_uncertainty(models, unlabeled_loader)
            arg = np.argsort(uncertainty)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(np.asarray(subset)[arg][-Config.addendum:])
            unlabeled_set = list(np.asarray(subset)[arg][:-Config.addendum]) + unlabeled_set[Config.subset:]

            # Create a new dataloader for the updated labeled dataset
            data_loaders['train'] = DataLoader(cifar10_train, batch_size=Config.batch_size,
                                               sampler=SubsetRandomSampler(labeled_set), pin_memory=True)
            pass

        # Save a checkpoint
        torch.save({'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()},
                   Tools.new_dir('./cifar10/resnet18_trial_{}.pth'.format(trial)))
        pass

    pass


class Config(object):
    num_train = 50000  # N
    batch_size = 128  # B
    subset = 10000  # M
    addendum = 1000  # K

    margin = 1.0  # xi
    weight = 1.0  # lambda

    trials = 3
    cycles = 10

    epoch = 200
    lr = 0.1
    milestones = [160]
    epoch_stop = 120  # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

    vis_step_count = 0
    # vis_vis = visdom.Visdom(server='http://localhost', port=9000)
    # vis_plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}
    vis_vis = None
    vis_plot_data = None

    if "Linux" in platform.platform():
        data_root = "/mnt/4T/Data/data/CIFAR"
    else:
        data_root = "D:\\Data\\cifar10"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pass


if __name__ == '__main__':
    runner = Runner(device=Config.device)
    for trial in range(Config.trials):
        runner.main(trial=trial)
        pass
    pass
