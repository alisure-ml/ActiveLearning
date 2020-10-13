import os
import math
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision.datasets import MNIST, SVHN, CIFAR10


class Utils(object):

    @staticmethod
    def gaussian_nll(mu, log_sigma, noise):
        NLL = torch.sum(log_sigma, 1) + \
              torch.sum(((noise - mu) / (1e-8 + torch.exp(log_sigma))) ** 2, 1) / 2.
        return NLL.mean()

    @staticmethod
    def schedule(p):
        return 2.0 / (1.0 + math.exp(- 10.0 * p)) - 1

    @staticmethod
    def numpy_to_variable(x):
        return Variable(torch.from_numpy(x).to(x.device))

    @staticmethod
    def log_sum_exp(logits, mask=None, inf=1e7):
        if mask is not None:
            logits = logits * mask - inf * (1.0 - mask)
            max_logits = logits.max(1, keepdim=True)[0]
            return ((logits - max_logits.expand_as(logits)).exp() * mask).sum(1,
                                                                              keepdim=True).log().squeeze() + max_logits.squeeze()
        else:
            max_logits = logits.max(1, keepdim=True)[0]
            return ((logits - max_logits.expand_as(logits)).exp()).sum(1,
                                                                       keepdim=True).log().squeeze() + max_logits.squeeze()

    @staticmethod
    def log_sum_exp_0(logits):
        max_logits = logits.max()
        return (logits - max_logits.expand_as(logits)).exp().sum().log() + max_logits

    @staticmethod
    def entropy(logits):
        probs = nn.functional.softmax(logits)
        ent = (- probs * logits).sum(1).squeeze() + log_sum_exp(logits)
        return ent.mean()

    @staticmethod
    def SumCELoss(logits, mask):
        log_sum_exp_all = log_sum_exp(logits, all_true)
        log_sum_exp_mask = log_sum_exp(logits, mask)
        return (- log_sum_exp_mask + log_sum_exp_all).mean()

    @staticmethod
    def one_hot(logits, labels):
        mask = Variable(torch.zeros(logits.size(0), logits.size(1)).to(logits.device))
        mask.data.scatter_(1, labels.data.view(-1, 1), 1)
        return mask

    @staticmethod
    def grad_norm(parameters, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm_type = float(norm_type)
        if norm_type == float('inf'):
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0
            for p in parameters:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        return total_norm

    @staticmethod
    def sample_z(mu, log_sigma):
        stds = (0.5 * log_sigma).exp().to(mu.device)
        epsilon = torch.randn(*mu.size()).to(mu.device)
        z = epsilon * stds + mu
        return z

    @staticmethod
    def save_model_by_name(model, global_step):
        save_dir = model.config.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, 'model-{:08d}.pt'.format(global_step))
        state = model.state_dict()
        torch.save(state, file_path)
        print('Saved to {}'.format(file_path))

    @staticmethod
    def load_model_by_name(model, dir_path, global_step, device='cuda'):
        file_path = os.path.join(dir_path, 'model-{:08d}.pt'.format(global_step))
        state = torch.load(file_path, map_location=device)
        model.load_state_dict(state)
        print("Loaded from {}".format(file_path))

    @staticmethod
    def query(mask, prefs, budget):
        current_indexes = list(np.flatnonzero(mask))
        prefs = list(prefs)
        candids = [ele for ele in prefs if ele not in current_indexes]
        new_indexes = candids[:budget]
        new_mask = np.zeros_like(mask, dtype=np.bool)
        new_mask[current_indexes + new_indexes] = True
        return new_mask

    pass


class DataLoader(object):

    def __init__(self, config, raw_loader, indices, batch_size):
        self.images, self.labels = [], []
        for idx in indices:
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)

        self.images = torch.stack(self.images, 0)
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.int64)).squeeze()

        if config.dataset == 'mnist':
            self.images = self.images.view(self.images.size(0), -1)

        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True)
        self.len = len(indices)

    def get_zca_cuda(self, reg=1e-6):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        mean = images.mean(0)
        images -= mean.expand_as(images)
        sigma = torch.mm(images.transpose(0, 1), images) / images.size(0)
        U, S, V = torch.svd(sigma)
        components = torch.mm(torch.mm(U, torch.diag(1.0 / torch.sqrt(S) + reg)), U.transpose(0, 1))
        return components, mean

    def apply_zca_cuda(self, components):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        self.images = torch.mm(images, components.transpose(0, 1)).cpu()

    def generator(self, inf=False):
        while True:
            indices = np.arange(self.images.size(0))
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices).long()
            for start in range(0, indices.size(0), self.batch_size):
                end = min(start + self.batch_size, indices.size(0))
                ret_images, ret_labels = self.images[indices[start: end]], self.labels[indices[start: end]]
                yield ret_images, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return self.len


def get_mnist_loaders(config):
    transform = transforms.Compose([transforms.ToTensor()])
    training_set = MNIST(config.data_root, train=True, download=True, transform=transform)
    dev_set = MNIST(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    if hasattr(config, 'mask_file'):
        mask = np.load(config.mask_file)
        print('mask loaded')
    else:
        for i in range(10):
            mask[np.where(labels == i)[0][: config.size_labeled_data // 10]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader


def get_svhn_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = SVHN(config.data_root, split='train', download=True, transform=transform)
    dev_set = SVHN(config.data_root, split='test', download=True, transform=transform)

    def preprocess(data_set):
        for i in range(len(data_set.data)):
            if data_set.labels[i] == 10:
                data_set.labels[i] = 0
    preprocess(training_set)
    preprocess(dev_set)

    indices = np.arange(len(training_set))
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    if hasattr(config, 'mask_file'):
        mask = np.load(config.mask_file)
        print('mask loaded')
    else:
        for i in range(10):
            mask[np.where(labels == i)[0][: config.size_labeled_data // 10]] = True
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    labeled_indices, unlabeled_indices = indices[mask], indices
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader


def get_cifar_loaders(config, mask_file):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR10(config.data_root, train=True, download=True, transform=transform)
    dev_set = CIFAR10(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    mask = np.load(mask_file)
    print('mask loaded')
    labeled_indices, unlabeled_indices = indices[mask], indices
    print('labeled size', labeled_indices.shape[0], 'unlabeled size',
          unlabeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)
    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader

