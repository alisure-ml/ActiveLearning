import time
import math
import torch
import random
import os, sys
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import cifar_model
import torch.nn.functional as F
import torchvision.utils as vutils
from collections import OrderedDict
from data.data import Utils, DataLoader, get_cifar_loaders


class Trainer(object):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config.data_config
        self.logger = open(self.config.log_path, 'a')
        self.device = self.config.device

        self.dis = cifar_model.Discriminative(config=self.data_config).to(self.device)
        self.gen = cifar_model.Generator(image_size=self.data_config.image_size,
                                         noise_size=self.data_config.noise_size).to(self.device)
        self.enc = cifar_model.Encoder(image_size=self.data_config.image_size,
                                       noise_size=self.data_config.noise_size, output_params=True).to(self.device)
        self.smp = cifar_model.Sampler(noise_size=self.data_config.noise_size).to(self.device)

        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=self.data_config.dis_lr, betas=(0.5, 0.999))
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.data_config.gen_lr, betas=(0.0, 0.999))
        self.enc_optimizer = optim.Adam(self.enc.parameters(), lr=self.data_config.enc_lr, betas=(0.0, 0.999))
        self.smp_optimizer = optim.Adam(self.smp.parameters(), lr=self.data_config.smp_lr, betas=(0.5, 0.9999))

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.d_criterion = nn.CrossEntropyLoss()

        iter = self.load_checkpoint(self.config.save_dir)
        if iter == 0 and self.config.inherit is not None:
            self.load_checkpoint(self.config.inherit_dir)
            self.iter_cnt = 0
            pass

        self.labeled_loader, self.unlabeled_loader, self.unlabeled_loader2, self.dev_loader = get_cifar_loaders(
            self.data_config, self.config.mask_file)
        pass

    def _train(self):
        self.dis.train()
        self.gen.train()
        self.enc.train()
        self.smp.train()

        ##### train Dis
        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = lab_images.to(self.device), lab_labels.to(self.device)

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = unl_images.to(self.device)

        noise = torch.rand(unl_images.size(0), self.data_config.noise_size).to(self.device)
        label = torch.randint(self.data_config.num_label, (unl_images.size(0),)).to(self.device)
        gen_images = self.gen(noise, label)

        lab_logits = self.dis(lab_images)
        unl_logits = self.dis(unl_images)
        gen_logits = self.dis(gen_images.detach())

        lab_loss = self.data_config.cls_lambda * self.d_criterion(lab_logits, lab_labels)

        unl_logsumexp = Utils.log_sum_exp(unl_logits)
        gen_logsumexp = Utils.log_sum_exp(gen_logits)

        true_loss = self.data_config.gan_lambda * \
                    (- 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp)))
        fake_loss = self.data_config.gan_lambda * (0.5 * torch.mean(F.softplus(gen_logsumexp)))
        unl_loss = true_loss + fake_loss

        d_loss = lab_loss + unl_loss

        self.dis_optimizer.zero_grad()
        d_loss.backward()
        self.dis_optimizer.step()

        ##### train Gen & Enc
        noise = torch.rand(unl_images.size(0), self.data_config.noise_size).to(self.device)
        label = torch.randint(self.data_config.num_label, (unl_images.size(0),)).to(self.device)
        gen_images = self.gen(noise, label)

        mu, log_sigma = self.enc(gen_images, label)
        vi_loss = Utils.gaussian_nll(mu, log_sigma, noise)

        unl_feat = self.dis(unl_images, feat=True)
        gen_feat = self.dis(gen_images, feat=True)
        g_loss = torch.mean(torch.abs(torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)))

        g_loss += self.data_config.trd_lambda * self.data_config.num_label * vi_loss

        self.gen_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()
        g_loss.backward()
        self.gen_optimizer.step()
        self.enc_optimizer.step()

        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = lab_images.to(self.device), lab_labels.to(self.device)

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = unl_images.to(self.device)
        unl_pls = self.dis(unl_images)

        lab_mu, lab_log_sigma = self.enc(lab_images, lab_labels)
        unl_mu, unl_log_sigma = self.enc(unl_images, unl_pls)

        z = Utils.sample_z(unl_mu, unl_log_sigma)
        unl_images_recon = self.gen(z, unl_pls).detach()
        rec = self.mse_loss(unl_images_recon, unl_images)
        kl = -0.5 * torch.sum(1 + unl_log_sigma - unl_mu.pow(2) - unl_log_sigma.exp())

        vae_loss = self.data_config.trd_lambda * (rec + kl)

        lab_preds = self.smp(lab_mu).view(-1)
        unl_preds = self.smp(unl_mu).view(-1)

        lab_real_preds = torch.ones(lab_images.size(0)).to(self.device)
        unl_real_preds = torch.ones(unl_images.size(0)).to(self.device)

        s_loss = self.data_config.adv_lambda * \
                 (self.bce_loss(lab_preds, lab_real_preds) + self.bce_loss(unl_preds, unl_real_preds))

        e_loss = vae_loss + s_loss
        ge_loss = g_loss + vae_loss + s_loss

        self.enc_optimizer.zero_grad()
        e_loss.backward()
        self.enc_optimizer.step()

        ##### train smp
        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = lab_images.to(self.device), lab_labels.to(self.device)

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = unl_images.to(self.device)
        unl_pls = self.dis(unl_images)

        lab_mu, _ = self.enc(lab_images, lab_labels)
        unl_mu, _ = self.enc(unl_images, unl_pls)

        lab_preds = self.smp(lab_mu).view(-1)
        unl_preds = self.smp(unl_mu).view(-1)

        lab_real_preds = torch.ones(lab_images.size(0)).to(self.device)
        unl_fake_preds = torch.zeros(unl_images.size(0)).to(self.device)

        s_loss = self.data_config.adv_lambda * \
                 (self.bce_loss(lab_preds, lab_real_preds) + self.bce_loss(unl_preds, unl_fake_preds))

        self.smp_optimizer.zero_grad()
        s_loss.backward()
        self.smp_optimizer.step()

        monitor_dict = OrderedDict([
            ('dis loss', d_loss.item()),
            ('gen & enc loss', ge_loss.item()),
            ('smp loss', s_loss.item())
        ])

        return monitor_dict

    def train(self):
        iter = self.iter_cnt
        monitor = OrderedDict()

        batch_per_epoch = int((len(self.unlabeled_loader) +
                               self.data_config.train_batch_size - 1) / self.data_config.train_batch_size)
        min_lr = self.data_config.min_lr if hasattr(self.data_config, 'min_lr') else 0.0
        while True:

            epoch = iter / batch_per_epoch
            if epoch >= self.data_config.max_epochs:
                break

            if iter % batch_per_epoch == 0:
                epoch_ratio = float(epoch) / float(self.data_config.max_epochs)
                self.dis_optimizer.param_groups[0]['lr'] = max(
                    min_lr, self.data_config.dis_lr * min(3. * (1. - epoch_ratio), 1.))
                self.gen_optimizer.param_groups[0]['lr'] = max(
                    min_lr, self.data_config.gen_lr * min(3. * (1. - epoch_ratio), 1.))
                self.enc_optimizer.param_groups[0]['lr'] = max(
                    min_lr, self.data_config.enc_lr * min(3. * (1. - epoch_ratio), 1.))
                self.smp_optimizer.param_groups[0]['lr'] = max(
                    min_lr, self.data_config.smp_lr * min(3. * (1. - epoch_ratio), 1.))

            iter_vals = self._train()

            for k, v in iter_vals.items():
                if k not in monitor:
                    monitor[k] = 0.
                monitor[k] += v

            if iter % self.data_config.save_period == 0:
                Utils.save_model_by_name(self, iter)

            if iter % self.data_config.eval_period == 0:
                train_loss, train_accuracy = self.eval(self.labeled_loader)
                dev_loss, dev_accuracy = self.eval(self.dev_loader)

                disp_str = '#{}-{}\ttrain: {:.4f}, {:.2f}% | dev: {:.4f}, {:.2f}%'.format(
                    int(epoch), iter, train_loss, train_accuracy * 100, dev_loss, dev_accuracy * 100)
                for k, v in monitor.items():
                    disp_str += ' | {}: {:.4f}'.format(k, v / self.data_config.eval_period)
                disp_str += '\n'

                monitor = OrderedDict()

                self.logger.write(disp_str)
                self.logger.flush()
                sys.stdout.write(disp_str)
                sys.stdout.flush()

            iter += 1
            self.iter_cnt += 1
            pass
        pass

    def eval(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        self.enc.eval()
        self.smp.eval()

        loss, correct, cnt = 0, 0, 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader.get_iter()):
                cnt += 1
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred_prob = self.dis(images)
                loss += self.d_criterion(pred_prob, labels).item()
                correct += torch.eq(torch.max(pred_prob, 1)[1], labels).data.sum()
                if max_batch is not None and i >= max_batch - 1: break
                pass
            pass

        return loss / cnt, correct.float() / len(data_loader)

    def eval2(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        self.enc.eval()
        self.smp.eval()

        lab_preds = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader.get_iter()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                mu, _ = self.enc(images, labels)
                pred_prob2 = self.smp(mu)
                lab_preds.extend(pred_prob2)
                if max_batch is not None and i >= max_batch - 1: break
            lab_preds = torch.stack(lab_preds)
            lab_preds = lab_preds.view(-1)
            lab_preds *= -1
        return lab_preds

    def load_checkpoint(self, dir_path):
        checkpoints = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
        if len(checkpoints) == 0:
            self.iter_cnt = 0
        else:
            last = max(checkpoints)
            self.iter_cnt = int(last[-11:-3])
            load_model_by_name(self, dir_path, self.iter_cnt)

        return self.iter_cnt

    pass

