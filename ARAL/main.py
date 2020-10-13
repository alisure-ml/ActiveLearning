import os
import torch
import platform
import numpy as np
from alisuretool.Tools import Tools
from trainer import svhn_trainer, mnist_trainer, cifar_trainer
from data.data import Utils, DataLoader, get_mnist_loaders, get_cifar_loaders, get_svhn_loaders


class ConfigMNIST(object):
    dataset = 'mnist'

    num_examples = 60000

    image_size = 28 * 28
    num_label = 10

    noise_size = 100

    gan_lambda = 1
    cls_lambda = 1
    trd_lambda = 0.001
    adv_lambda = 0.001

    dis_lr = 3e-3
    enc_lr = 1e-3
    gen_lr = 1e-3
    smp_lr = 3e-3

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    max_epochs = 2000

    eval_period = 500
    save_period = 10000

    if "Linux" in platform.platform():
        data_root = "/mnt/4T/Data/data/CIFAR"
    else:
        data_root = "D:\\Data\\cifar10"

    pass


class ConfigSVHN(object):
    dataset = 'svhn'

    num_examples = 73257

    image_size = 3 * 32 * 32
    num_label = 10

    noise_size = 100

    gan_lambda = 1
    cls_lambda = 1
    trd_lambda = 0.001
    adv_lambda = 0.001

    dis_lr = 1e-3
    enc_lr = 1e-3
    gen_lr = 1e-3
    smp_lr = 1e-3
    min_lr = 1e-4

    train_batch_size = 64
    train_batch_size_2 = 64
    dev_batch_size = 200

    max_epochs = 900

    eval_period = 500
    save_period = 10000

    if "Linux" in platform.platform():
        data_root = "/mnt/4T/Data/data/CIFAR"
    else:
        data_root = "D:\\Data\\cifar10"

    pass


class ConfigCIFAR(object):
    dataset = 'cifar'

    num_examples = 50000

    image_size = 3 * 32 * 32
    num_label = 10

    noise_size = 100

    gan_lambda = 1
    cls_lambda = 1
    trd_lambda = 0.001
    adv_lambda = 0.001

    dis_lr = 6e-4
    enc_lr = 3e-4
    gen_lr = 3e-4
    smp_lr = 6e-4

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    max_epochs = 1200

    eval_period = 500
    save_period = 10000

    if "Linux" in platform.platform():
        data_root = "/mnt/4T/Data/data/CIFAR"
    else:
        data_root = "D:\\Data\\cifar10"

    pass


class Config(object):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    budget = 100
    max_iterations = 10

    dataset = "cifar"
    if dataset == 'mnist':
        data_config = ConfigMNIST()
        Trainer = mnist_trainer.Trainer
    elif dataset == 'svhn':
        data_config = ConfigSVHN()
        Trainer = svhn_trainer.Trainer
    elif dataset == 'cifar':
        data_config = ConfigCIFAR()
        Trainer = cifar_trainer.Trainer
    else:
        raise NotImplementedError

    suffix = None
    inherit = None
    save_dir = None
    log_path = None
    inherit_dir = None
    log_root = Tools.new_dir(os.path.join("logs", '{}_{}_{}'.format(dataset, budget, "run0")))

    mask_file = None
    pass


def deal_mask(before_mask, mask_file, preds, budget):
    if os.path.exists(mask_file):
        mask = np.load(mask_file)
        Tools.print('{} loaded'.format(mask_file))
    else:
        mask = Utils.query(before_mask, preds, budget)
        np.save(mask_file, mask)
        Tools.print('{} saved'.format(mask_file))
    return mask


if __name__ == '__main__':
    Tools.print('------------------------- Iteration 1 -------------------------')

    Config.suffix = "iter1"
    Config.mask_file = os.path.join(Config.log_root, 'mask_{}_{}.npy'.format(Config.dataset, Config.suffix))
    mask = np.zeros(Config.data_config.num_examples, dtype=np.bool)
    mask = deal_mask(mask, Config.mask_file, np.random.permutation(Config.data_config.num_examples), Config.budget)

    Config.save_dir = Tools.new_dir('{}/{}_{}'.format(Config.log_root, Config.dataset, Config.suffix))
    Config.log_path = os.path.join(Config.save_dir, '{}_{}_log.txt'.format(Config.dataset, Config.suffix))

    model = Config.Trainer(Config)
    model.train()
    
    for i in range(Config.max_iterations):
        Tools.print('------------------------- Iteration {} -------------------------'.format(i + 2))

        Config.suffix = 'iter{}'.format(i + 2)
        Config.inherit = 'iter{}'.format(i + 1)
        Config.mask_file = os.path.join(Config.log_root, 'mask_{}_{}.npy'.format(Config.dataset, Config.suffix))
        mask = deal_mask(mask, Config.mask_file, None if os.path.exists(mask_file) else np.Configort(
            model.eval2(model.unlabeled_loader2).cpu()), Config.budget)

        Config.save_dir = Tools.new_dir('{}/{}_{}'.format(Config.log_root, Config.dataset, Config.suffix))
        Config.log_path = os.path.join(Config.save_dir, '{}_{}_log.txt'.format(Config.dataset, Config.suffix))
        Config.inherit_dir = Tools.new_dir('{}/{}_{}'.format(Config.log_root, Config.dataset, Config.inherit))

        model = Config.Trainer(Config)
        model.train()
        pass

    pass
    