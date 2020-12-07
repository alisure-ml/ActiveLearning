import os
import time
import tqdm
import torch
import platform
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Averager(object):

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    pass


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
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
        pass


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

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self,block=BasicBlock, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
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
            pass
        pass

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion))
            pass
        layers = [block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size)]
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)
    pass


class DeepEMD(nn.Module):

    def __init__(self, num_class=64):
        super().__init__()
        self.encoder = ResNet()
        self.fc = nn.Linear(640, num_class)
        pass

    def forward(self, input, mode=None):
        if mode == 'meta':
            support, query = input
            return self.emd_forward_1shot(support, query)
        elif mode == 'encoder':
            return self.encode(input, True)
        return self.pre_train_forward(input)

    def pre_train_forward(self, input):
        return self.fc(self.encode(input, dense=False).squeeze(-1).squeeze(-1))

    def emd_forward_1shot(self, proto, query):
        proto = proto.squeeze(0)

        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        proto = proto - proto.mean(1).unsqueeze(1)
        query = query - query.mean(1).unsqueeze(1)

        similarity_map = self.get_similiarity_map(proto, query)
        if not self.training:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2)
        return logits

    def get_weight_vector(self, A, B):
        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        proto = proto.unsqueeze(-3)
        query = query.unsqueeze(-2)
        query = query.repeat(1, 1, 1, feature_size, 1)
        similarity_map = F.cosine_similarity(proto, query, dim=-1)

        return similarity_map

    def get_emd_distance(self, similarity_map, weight_1, weight_2, temperature=12.5):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        for i in range(num_query):
            for j in range(num_proto):
                _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
                similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()

        temperature = temperature / num_node
        logitis = similarity_map.sum(-1).sum(-1) * temperature
        return logitis

    def encode(self, x, dense=True):
        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return x
        else:
            x = self.encoder(x)
            if not dense:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
        return x

    pass


class CategoriesSampler(object):
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
        pass

    pass


class MiniImageNet(Dataset):

    def __init__(self, setname, data_dir):
        csv_path = osp.join(data_dir, 'split', setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data, self.label, self.wnids = [], [], []
        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(data_dir, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
            self.data.append(path)
            self.label.append(self.wnids.index(wnid))
            pass
        self.num_class = len(set(self.label))

        if setname == 'val' or setname == 'test':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif setname == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
            pass

        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    pass


class Config(object):

    def __init__(self):
        self.gpu_id = 2
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.bs = 128
        self.lr = 0.1
        self.step_size = 30
        self.gamma = 0.2

        self.way = 5
        self.shot = 1
        self.query = 15
        self.num_episode = 100
        self.max_epoch = 100

        self.val_set = "val"

        self.num_class = 64
        self.save_all = False

        self.dataset = "miniimagenet"
        self.data_root = self.get_data_root(self.dataset)

        self.save_path = Tools.new_dir(os.path.join('checkpoint', 'pre_train/{}/{}-{:.4f}-{}-{:.2f}'.format(
            self.dataset, self.bs, self.lr, self.step_size, self.gamma)))

        self.set_seed(0)
        pass

    @staticmethod
    def get_data_root(dataset):
        if "Linux" in platform.platform():
            data_root = '/mnt/4T/Data/data/miniImagenet'
            if not os.path.isdir(data_root):
                data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
        else:
            data_root = "F:\\data\\miniImagenet"
        return data_root

    @staticmethod
    def set_seed(seed):
        if seed == 0:
            print(' random seed')
            torch.backends.cudnn.benchmark = True
        else:
            print('manual seed:', seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            pass
        pass

    pass


config = Config()


class RunnerPreTrain(object):

    def __init__(self):
        self.train_set = MiniImageNet('train', config.data_root)
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=config.bs,
                                       shuffle=True, num_workers=8, pin_memory=True)

        self.val_set = MiniImageNet(config.val_set, config.data_root)
        self.val_sampler = CategoriesSampler(self.val_set.label, self.num_episode, self.way, self.shot + self.query)
        self.val_loader = DataLoader(self.val_set, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)

        self.model = DeepEMD(num_class=config.num_class).cuda()
        self.label = torch.arange(config.way, dtype=torch.int8).repeat(config.query).type(torch.LongTensor).cuda()

        self.optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': config.lr},
                                          {'params': self.model.module.fc.parameters(), 'lr': config.lr}],
                                         momentum=0.9, nesterov=True, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config.step_size, gamma=config.gamma)
        pass

    def train(self):
        max_acc = 0.0
        max_acc_epoch = 0
        for epoch in range(1, config.max_epoch + 1):
            Tools.print(config.save_path)

            tl, ta = self.train_epoch()
            vl, va = self.test()

            Tools.print('epoch:{} lr:{:.4f} train:{:.4f} {:.4f}, val:{:.4f} {:.4f}, max:{:.4f}({:.4f})'.format(
                epoch, self.lr_scheduler.get_last_lr(), tl, ta, vl, va, max_acc, max_acc_epoch))

            if va >= max_acc:
                Tools.print('A better model is found: {} {}'.format(va))
                max_acc = va
                max_acc_epoch = epoch
                torch.save(optimizer.state_dict(), os.path.join(config.save_path, 'optimizer_best.pth'))
                self.save_model(max_acc)
                pass
            pass
        pass

    def train_epoch(self):
        tl, ta = Averager(), Averager()

        self.model.train()
        tqdm_gen = tqdm.tqdm(train_loader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, train_label = [_.cuda() for _ in batch]
            logits = self.model.forward(data)
            loss = F.cross_entropy(logits, train_label)
            acc = self.count_acc(logits, train_label)

            tqdm_gen.set_description('epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, loss.item(), acc))
            tl.add(loss.item())
            ta.add(acc)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pass
        self.lr_scheduler.step()
        return tl.item(), ta.item()

    def test(self):
        self.model.eval()
        vl, va = Averager(), Averager()
        with torch.no_grad():
            tqdm_gen = tqdm.tqdm(val_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, _ = [_.cuda() for _ in batch]
                k = config.way * config.shot
                data = self.model.forward(data, mode="encoder")
                data_shot, data_query = data[:k], data[k:]
                # episode learning
                model.module.mode = 'meta'
                logits = self.model.forward((data_shot, data_query), mode="meta")
                loss = F.cross_entropy(logits, label)
                acc = self.count_acc(logits, label)
                vl.add(loss.item())
                va.add(acc)
                tqdm_gen.set_description('epo {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
                pass
            pass
        return vl.item(), va.item()

    def save_model(self, name):
        torch.save(dict(params=self.model.module.encoder.state_dict()),
                   os.path.join(config.save_path, str(name) + '.pth'))
        pass

    @staticmethod
    def count_acc(logits, label):
        pred = torch.argmax(logits, dim=1)
        return (pred == label).cpu().mean().item()

    pass


if __name__ == '__main__':

    RunnerPreTrain()

    pass
