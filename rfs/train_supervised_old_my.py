import os
import sys
import time
import scipy
import torch
import pickle
import argparse
import platform
import warnings
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import t
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.distributions import Bernoulli
import torchvision.transforms as transforms
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class TransformMy(object):

    @staticmethod
    def train():
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose([lambda x: Image.fromarray(x), transforms.RandomCrop(84, padding=8),
                                   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                   transforms.RandomHorizontalFlip(), lambda x: np.asarray(x),
                                   transforms.ToTensor(), normalize])

    @staticmethod
    def test():
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        return transforms.Compose([lambda x: Image.fromarray(x), transforms.ToTensor(), normalize])

    pass


class ImageNet(Dataset):

    def __init__(self, data_root, partition='train', pretrain=True, transform=None):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.partition = partition
        self.pretrain = pretrain
        self.transform = transform

        if self.pretrain:
            self.file_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'
        else:
            self.file_pattern = 'miniImageNet_category_split_%s.pickle'

        self.data = {}
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels = data['labels']
            pass
        pass

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)
        return img, target, item

    def __len__(self):
        return len(self.labels)

    pass


class MetaImageNet(ImageNet):

    def __init__(self, data_root, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaImageNet, self).__init__(data_root, partition,pretrain=False)
        self.fix_seed = fix_seed
        self.n_ways = Config.n_ways
        self.n_shots = Config.n_shots
        self.n_queries = Config.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = Config.n_test_runs
        self.n_aug_support_samples = Config.n_aug_support_samples
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())
        pass

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs

    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    pass


class ValFewShot(object):

    @staticmethod
    def proto(support, support_ys, query):
        """Protonet classifier"""
        nc = support.shape[-1]
        support = np.reshape(support, (-1, 1, Config.n_ways, Config.n_shots, nc))
        support = support.mean(axis=3)
        batch_size = support.shape[0]
        query = np.reshape(query, (batch_size, -1, 1, nc))
        logits = - ((query - support) ** 2).sum(-1)
        pred = np.argmax(logits, axis=-1)
        pred = np.reshape(pred, (-1,))
        return pred

    @staticmethod
    def nn(support, support_ys, query):
        """nearest classifier"""
        support = np.expand_dims(support.transpose(), 0)
        query = np.expand_dims(query, 2)

        diff = np.multiply(query - support, query - support)
        distance = diff.sum(1)
        min_idx = np.argmin(distance, axis=1)
        pred = [support_ys[idx] for idx in min_idx]
        return pred

    @staticmethod
    def cosine(support, support_ys, query):
        """Cosine classifier"""
        support_norm = np.linalg.norm(support, axis=1, keepdims=True)
        support = support / support_norm
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / query_norm

        cosine_distance = query @ support.transpose()
        max_idx = np.argmax(cosine_distance, axis=1)
        pred = [support_ys[idx] for idx in max_idx]
        return pred

    pass


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class DropBlock(nn.Module):

    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask

    pass


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(planes, 4)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

    pass


class ResNet(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, num_classes=-1, use_se=False):
        super(ResNet, self).__init__()

        self.inplanes = 3
        self.use_se = use_se
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        f0 = x
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        f3 = x
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x
        if self.num_classes > 0:
            x = self.classifier(x)

        if is_feat:
            return [f0, f1, f2, f3, feat], x
        else:
            return x
        pass

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class Runner(object):

    def __init__(self):
        self.train_loader = DataLoader(
            ImageNet(data_root=Config.data_root, partition='train', transform=TransformMy.train()),
            batch_size=Config.batch_size, shuffle=True, drop_last=True, num_workers=Config.num_workers)
        self.val_loader = DataLoader(
            ImageNet(data_root=Config.data_root, partition='val', transform=TransformMy.test()),
            batch_size=Config.batch_size // 2, shuffle=False, drop_last=False, num_workers=Config.num_workers // 2)

        self.meta_trainloader = DataLoader(
            MetaImageNet(data_root=Config.data_root, partition='train_phase_test',
                         train_transform=TransformMy.train(), test_transform=TransformMy.test(), fix_seed=False),
            batch_size=Config.test_batch_size, shuffle=False, drop_last=False, num_workers=Config.num_workers)
        self.meta_testloader = DataLoader(
            MetaImageNet(data_root=Config.data_root, partition='test',
                         train_transform=TransformMy.train(), test_transform=TransformMy.test(), fix_seed=False),
            batch_size=Config.test_batch_size, shuffle=False, drop_last=False, num_workers=Config.num_workers)
        self.meta_valloader = DataLoader(
            MetaImageNet(data_root=Config.data_root, partition='val',
                         train_transform=TransformMy.train(), test_transform=TransformMy.test(), fix_seed=False),
            batch_size=Config.test_batch_size, shuffle=False, drop_last=False, num_workers=Config.num_workers)

        # model
        self.model = ResNet(BasicBlock, [1, 1, 1, 1], avg_pool=True,
                            drop_rate=0.1, dropblock_size=5, num_classes=64).cuda()

        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss().cuda()
        pass

    def train(self):
        # routine: supervised pre-training
        for epoch in range(1, Config.epochs + 1):
            self.adjust_learning_rate(epoch)
            print("==> training...")

            time1 = time.time()
            train_acc, train_loss = self.train_epoch(epoch)
            time2 = time.time()
            print('epoch {}, {} {}, total time {:.2f}'.format(epoch, train_acc, train_loss, time2 - time1))

            test_acc, test_acc_top5, test_loss = self.validate()
            print('epoch {}, {} {} {}'.format(epoch, test_acc, test_acc_top5, test_loss))

            # regular saving
            if epoch % Config.save_freq == 0:
                print('==> Saving...')
                state = {'epoch': epoch, 'model': self.model.state_dict()}
                save_file = os.path.join(Config.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

                self.validate_few_shot(classifier=Config.classifier)
                pass
            pass

        # save the last model
        torch.save(self.model.state_dict(), os.path.join(Config.model_path, 'last.pth'))

        self.validate_few_shot(classifier=Config.classifier)
        pass

    def train_epoch(self, epoch):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for idx, (input, target, _) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            input = input.float().cuda()
            target = target.cuda()

            # ===================forward=====================
            output = self.model(input)
            loss = self.criterion(output, target)

            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # ===================backward=====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if idx % Config.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg, losses.avg

    def validate(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for idx, (input, target, _) in enumerate(self.val_loader):
                input = input.float().cuda()
                target = target.cuda()

                # compute output
                output = self.model(input)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % Config.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        idx, len(self.val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg, top5.avg, losses.avg

    def validate_few_shot(self, classifier='Proto'):
        start = time.time()
        val_acc, val_std = self.meta_test(self.meta_valloader, classifier=classifier)
        val_time = time.time() - start
        print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std, val_time))

        start = time.time()
        val_acc_feat, val_std_feat = self.meta_test(self.meta_valloader, use_logit=False, classifier=classifier)
        val_time = time.time() - start
        print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat, val_std_feat, val_time))

        start = time.time()
        test_acc, test_std = self.meta_test(self.meta_testloader, classifier=classifier)
        test_time = time.time() - start
        print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

        start = time.time()
        test_acc_feat, test_std_feat = self.meta_test(self.meta_testloader, use_logit=False, classifier=classifier)
        test_time = time.time() - start
        print('test_acc_feat:{:.4f},test_std:{:.4f},time:{:.1f}'.format(test_acc_feat, test_std_feat, test_time))
        pass

    def meta_test(self, testloader, use_logit=True, is_norm=True, classifier='LR'):
        self.model.eval()
        acc = []

        with torch.no_grad():
            for idx, data in tqdm(enumerate(testloader)):
                support_xs, support_ys, query_xs, query_ys = data
                support_xs = support_xs.cuda()
                query_xs = query_xs.cuda()
                batch_size, _, channel, height, width = support_xs.size()
                support_xs = support_xs.view(-1, channel, height, width)
                query_xs = query_xs.view(-1, channel, height, width)

                if use_logit:
                    support_features = self.model(support_xs).view(support_xs.size(0), -1)
                    query_features = self.model(query_xs).view(query_xs.size(0), -1)
                else:
                    feat_support, _ = self.model(support_xs, is_feat=True)
                    support_features = feat_support[-1].view(support_xs.size(0), -1)
                    feat_query, _ = self.model(query_xs, is_feat=True)
                    query_features = feat_query[-1].view(query_xs.size(0), -1)

                if is_norm:
                    support_features = self.normalize(support_features)
                    query_features = self.normalize(query_features)
                    pass

                support_features = support_features.detach().cpu().numpy()
                query_features = query_features.detach().cpu().numpy()

                support_ys = support_ys.view(-1).numpy()
                query_ys = query_ys.view(-1).numpy()

                #  clf = SVC(gamma='auto', C=0.1)
                if classifier == 'LR':
                    clf = LogisticRegression(penalty='l2', random_state=0, C=1.0,
                                             solver='lbfgs', max_iter=1000, multi_class='multinomial')
                    clf.fit(support_features, support_ys)
                    query_ys_pred = clf.predict(query_features)
                elif classifier == 'SVM':
                    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1, kernel='linear',
                                                              decision_function_shape='ovr'))
                    clf.fit(support_features, support_ys)
                    query_ys_pred = clf.predict(query_features)
                elif classifier == 'NN':
                    query_ys_pred = ValFewShot.nn(support_features, support_ys, query_features)
                elif classifier == 'Cosine':
                    query_ys_pred = ValFewShot.cosine(support_features, support_ys, query_features)
                elif classifier == 'Proto':
                    query_ys_pred = ValFewShot.proto(support_features, support_ys, query_features)
                else:
                    raise NotImplementedError('classifier not supported: {}'.format(classifier))

                acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
            pass

        return self.mean_confidence_interval(acc)

    @staticmethod
    def normalize(x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * t._ppf((1 + confidence) / 2., n - 1)
        return m, h

    def adjust_learning_rate(self, epoch):
        steps = np.sum(epoch > np.asarray(Config.lr_decay_epochs))
        if steps > 0:
            new_lr = Config.learning_rate * (Config.lr_decay_rate ** steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            pass
        pass

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        pass

    pass


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore')
    cudnn.benchmark = True

    print_freq = 100
    save_freq = 5
    batch_size = 64
    num_workers = 8
    epochs = 100

    n_test_runs = 600
    n_ways = 5
    n_shots = 1
    n_queries = 15
    n_aug_support_samples = 5
    test_batch_size = 1

    # classifier = 'LR'
    classifier = 'Proto'
    # classifier = 'NN'
    # classifier = 'Cosine'
    # classifier = 'SVM'

    learning_rate = 0.05
    lr_decay_epochs = [60, 80]
    lr_decay_rate = 0.1

    model_name = "{}".format(classifier)
    model_path = Tools.new_dir('./models_pretrained/{}'.format(model_name))

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet/miniImageNet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet/miniImageNet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pass


"""
# Proto
epoch 100, 99.38802337646484 0.04409980621809761, total time 76.99
epoch 100, 83.74759674072266 95.8822250366211 0.6583566182879769
val_acc: 0.6540, val_std: 0.0073, time: 40.0
val_acc_feat: 0.6545, val_std: 0.0075, time: 39.7
test_acc: 0.6161, test_std: 0.0083, time: 39.0
test_acc_feat:0.5968,test_std:0.0081,time:39.3
val_acc: 0.6530, val_std: 0.0073, time: 39.0
val_acc_feat: 0.6562, val_std: 0.0076, time: 39.5
test_acc: 0.6167, test_std: 0.0082, time: 39.2
test_acc_feat:0.5944,test_std:0.0083,time:39.4

# LR
epoch 100, 99.3515625 0.045092997271567584, total time 72.31
epoch 100, 83.10752868652344 95.68487548828125 0.6670569186465481
val_acc: 0.6445, val_std: 0.0088, time: 40.3
val_acc_feat: 0.6543, val_std: 0.0085, time: 43.7
test_acc: 0.6134, test_std: 0.0088, time: 40.6
test_acc_feat:0.6163,test_std:0.0082,time:43.9
val_acc: 0.6446, val_std: 0.0089, time: 40.4
val_acc_feat: 0.6536, val_std: 0.0086, time: 43.7
test_acc: 0.6129, test_std: 0.0089, time: 40.2
test_acc_feat:0.6162,test_std:0.0081,time:43.3

# Cosine
epoch 100, 99.42448425292969 0.04334269272784392, total time 73.98
epoch 100, 83.41156768798828 95.84488677978516 0.6576445659219977
val_acc: 0.6398, val_std: 0.0094, time: 38.1
val_acc_feat: 0.6323, val_std: 0.0089, time: 38.4
test_acc: 0.6031, test_std: 0.0079, time: 38.2
test_acc_feat:0.5975,test_std:0.0076,time:38.7
val_acc: 0.6397, val_std: 0.0093, time: 38.5
val_acc_feat: 0.6315, val_std: 0.0090, time: 38.5
test_acc: 0.6038, test_std: 0.0080, time: 38.4
test_acc_feat:0.5972,test_std:0.0075,time:38.7
"""


if __name__ == '__main__':
    runner = Runner()
    runner.train()
