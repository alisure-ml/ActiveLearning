import os
import sys
import time
import torch
import socket
import argparse
import platform
import warnings
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from alisuretool.Tools import Tools
from eval.meta_eval import meta_test
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.resnet import resnet18, resnet12
from dataset.mini_imagenet import ImageNet, MetaImageNet
from util import adjust_learning_rate, accuracy, AverageMeter


class Config(object):
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore')
    cudnn.benchmark = True

    print_freq = 100
    save_freq = 10
    batch_size = 64
    num_workers = 8
    epochs = 100

    learning_rate = 0.05
    lr_decay_epochs = [60, 80]
    lr_decay_rate = 0.1

    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/miniImagenet'
        if not os.path.isdir(data_root):
            data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
    else:
        data_root = "F:\\data\\miniImagenet"

    pass


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # specify folder
    parser.add_argument('--model_path', type=str, default='results/models', help='path to save model')
    parser.add_argument('--data_root', type=str, default=Config.data_root, help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='test batch')

    opt = parser.parse_args()

    opt.data_root = '{}/miniImageNet'.format(opt.data_root)
    opt.data_aug = True
    opt.model_name = 'my'
    opt.save_folder = Tools.new_dir(os.path.join(opt.model_path, opt.model_name))
    return opt


class Runner(object):

    def __init__(self, opt):
        self.opt = opt
        self.n_cls = 64
        self.train_trans, self.test_trans = self.transforms_options()
        self.train_loader = DataLoader(ImageNet(args=self.opt, partition='train', transform=self.train_trans),
                                       batch_size=Config.batch_size, shuffle=True, drop_last=True,
                                       num_workers=Config.num_workers)
        self.val_loader = DataLoader(ImageNet(args=self.opt, partition='val', transform=self.test_trans),
                                     batch_size=Config.batch_size // 2, shuffle=False, drop_last=False,
                                     num_workers=Config.num_workers // 2)

        self.meta_trainloader = DataLoader(MetaImageNet(args=self.opt, partition='train_phase_test',
                                                        train_transform=self.train_trans,
                                                        test_transform=self.test_trans, fix_seed=False),
                                           batch_size=self.opt.test_batch_size, shuffle=False, drop_last=False,
                                           num_workers=Config.num_workers)
        self.meta_valloader = DataLoader(MetaImageNet(args=self.opt, partition='val', train_transform=self.train_trans,
                                                      test_transform=self.test_trans, fix_seed=False),
                                         batch_size=self.opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=Config.num_workers)
        self.meta_testloader = DataLoader(MetaImageNet(args=self.opt, partition='test', train_transform=self.train_trans,
                                                       test_transform=self.test_trans, fix_seed=False),
                                          batch_size=self.opt.test_batch_size, shuffle=False, drop_last=False,
                                          num_workers=Config.num_workers)

        # model
        self.model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=self.n_cls).cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss().cuda()
        pass

    def train(self):
        for epoch in range(1, Config.epochs + 1):
            self.adjust_learning_rate(epoch)
            print("==> training...")

            time1 = time.time()
            train_acc, train_loss = self.train_epoch(epoch)
            time2 = time.time()
            print('epoch {}, total time {:.2f} acc {:.4f} loss {:.4f}'.format(
                epoch, time2 - time1, train_acc, train_loss))

            test_acc, test_acc_top5, test_loss = self.validate()
            print('epoch {}, acc {:.4f} acc5 {:.4f} loss {:.4f}'.format(
                epoch, test_acc, test_acc_top5, train_loss))

            if epoch % 5 == 0:
                self.validate_few_shot()

            # regular saving
            if epoch % Config.save_freq == 0:
                print('==> Saving...')
                torch.save({'epoch': epoch, 'model': self.model.state_dict()},
                           os.path.join(self.opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)))
                pass
            pass

        # save the last model
        torch.save({'opt': self.opt, 'model': self.model.state_dict()},
                   os.path.join(self.opt.save_folder, 'last.pth'))

        self.validate_few_shot()
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

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
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
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
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
                        idx, len(self.val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg, top5.avg, losses.avg

    def validate_few_shot(self):
        # start = time.time()
        # train_acc, train_std = meta_test(self.model, self.meta_trainloader)
        # train_time = time.time() - start
        # print('train_acc: {:.4f}, train_std: {:.4f}, time: {:.1f}'.format(train_acc, train_std, train_time))
        #
        # start = time.time()
        # train_acc_feat, train_std_feat = meta_test(self.model, self.meta_trainloader, use_logit=False)
        # train_time = time.time() - start
        # print('train_acc_feat: {:.4f}, train_std: {:.4f}, time: {:.1f}'.format(
        #     train_acc_feat, train_std_feat, train_time))

        start = time.time()
        val_acc, val_std = meta_test(self.model, self.meta_valloader)
        val_time = time.time() - start
        print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std, val_time))

        start = time.time()
        val_acc_feat, val_std_feat = meta_test(self.model, self.meta_valloader, use_logit=False)
        val_time = time.time() - start
        print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat, val_std_feat, val_time))

        start = time.time()
        test_acc, test_std = meta_test(self.model, self.meta_testloader)
        test_time = time.time() - start
        print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

        start = time.time()
        test_acc_feat, test_std_feat = meta_test(self.model, self.meta_testloader, use_logit=False)
        test_time = time.time() - start
        print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(
            test_acc_feat, test_std_feat, test_time))
        pass

    @staticmethod
    def transforms_options():
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        train = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
        test = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            normalize
        ])
        return train, test

    def adjust_learning_rate(self, epoch):
        steps = np.sum(epoch > np.asarray(Config.lr_decay_epochs))
        if steps > 0:
            new_lr = Config.learning_rate * (Config.lr_decay_rate ** steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            pass
        pass
    pass


"""
 * Acc@1 99.451 Acc@5 99.995
 * Acc@1 83.358 Acc@5 95.882
 
 * Acc@1 99.349 Acc@5 99.995
 * Acc@1 83.369 Acc@5 95.877
 
 k=1
val_acc: 0.6512, val_std: 0.0087, time: 41.5
val_acc_feat: 0.6493, val_std: 0.0087, time: 44.7
test_acc: 0.6064, test_std: 0.0080, time: 41.7
test_acc_feat: 0.6100, test_std: 0.0081, time: 44.9

k=5
train_acc: 0.9375, train_std: 0.0033, time: 76.0
train_acc_feat: 0.9408, train_std: 0.0032, time: 98.0
val_acc: 0.7959, val_std: 0.0058, time: 75.9
val_acc_feat: 0.8120, val_std: 0.0055, time: 99.0
test_acc: 0.7669, test_std: 0.0066, time: 76.4
test_acc_feat: 0.7894, test_std: 0.0063, time: 100.5
"""


if __name__ == '__main__':
    runner = Runner(parse_option())
    runner.train()
