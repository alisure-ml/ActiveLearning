import os
import cv2
import glob
import torch
import random
import platform
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from alisuretool.Tools import Tools
from torchvision.models import alexnet
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class Normalize(object):

    def __init__(self, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
        self.mean = mean
        self.std = std
        pass

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = img - self.mean
        img /= self.std
        sample = {'image': img, 'label': label}
        return sample

    pass


class SquarifyImage(object):

    """
    Scale and squarify an image into box of fixed ize
    """

    def __init__(self, box_size=256, scale=(0.6, 1.2), is_scale=True, seed=None):
        super(SquarifyImage, self).__init__()
        self.box_size = box_size
        self.min_scale_ratio = scale[0]
        self.max_scale_ratio = scale[1]
        self.is_scale = is_scale
        self.seed = seed
        pass

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = self.squarify(img)
        sample = {'image': img, 'label': label}
        return sample

    def squarify(self, img):
        """
        Squarfiy the image
        Parameters
        ----------
        img : np.ndarray
            1-channel or 3-channels image

        Returns
        -------
        img_padded : np.ndarray
        """
        if self.is_scale:
            img_scaled = self.img_scale(img)
            img = img_scaled
        w, h, _ = img.shape

        ratio = min(self.box_size / w, self.box_size / h)
        resize_w, resize_h = int(w * ratio), int(h * ratio)
        x_pad, y_pad = (self.box_size - resize_w) // 2, (self.box_size - resize_h) // 2
        t_pad, b_pad = x_pad, self.box_size - resize_w - x_pad
        l_pad, r_pad = y_pad, self.box_size - resize_h - y_pad

        resized_img = cv2.resize(img, (resize_h, resize_w))

        img_padded = cv2.copyMakeBorder(resized_img, top=t_pad, bottom=b_pad,
                                        left=l_pad, right=r_pad, borderType=0, value=0)

        if img_padded.shape == [self.box_size, self.box_size, 3]:
            raise ValueError('Invalid size for squarified image {} !'.format(img_padded.shape))
        return img_padded

    def img_scale(self, img):
        """
        Randomly scaling an image
        Parameters
        ----------
        img  : np.ndarray
            1-channel or 3-channels image

        Returns
        -------
        img_scaled : np.ndarray
        """
        scale = np.random.uniform(self.min_scale_ratio, self.max_scale_ratio, self.seed)
        img_scaled = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        return img_scaled

    pass


class RandomCrop(object):

    def __init__(self, target_size):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            assert len(target_size) == 2
            self.target_size = target_size
        pass

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        h, w = img.shape[:2]
        new_h, new_w = self.target_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h, left: left + new_w]
        sample = {'image': img, 'label': label}
        return sample

    pass


class ToTensor(object):

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = img.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(img), 'label': label}
        return sample

    pass


class CalTech256Split(Dataset):

    def __init__(self, root_dir, classes_num=256, split_test=0.0):
        self.root_dir = root_dir
        self.classes_num = classes_num

        self.data_train, self.label_train, self.data_test, self.label_test = [], [], [], []
        for cat in range(0, self.classes_num):
            cat_dir = glob.glob(os.path.join(self.root_dir, '%03d*' % (cat + 1)))[0]
            img_files = glob.glob(os.path.join(cat_dir, '*.jpg'))
            random.shuffle(img_files)
            test_num = int(len(img_files) * split_test)

            self.data_test.extend(img_files[0: test_num])
            self.label_test.extend([cat] * test_num)
            self.data_train.extend(img_files[test_num:])
            self.label_train.extend([cat] * (len(img_files) - test_num))
            pass

        pass

    pass


class CalTech256Dataset(Dataset):

    def __init__(self, transform, data, label, classes_num=256):
        self.data = data
        self.label = label
        self.transform = transform
        self.classes_num = classes_num
        pass

    def __getitem__(self, idx):
        img, label = self.data[idx], self.label[idx]
        img = cv2.imread(img)
        img = img[:, :, ::-1] / 255.0
        sample = {'image': img, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)

    pass


class Criteria(object):

    @staticmethod
    def update_threshold(delta, dr, t):
        if t > 0:
            delta = delta - dr * t
        return delta

    @staticmethod
    def least_confidence(pred_prob, k):
        """
        Rank all the unlabeled samples in an ascending order according to
        equation 2
    
        Parameters
        ----------
        pred_prob : prediction probability of x_i with dimension (batch x n_class)
        k : int
            most informative samples
        Returns
        -------
        np.array with dimension (K x 1) containing the indices of the K
            most informative samples.
        np.array with dimension (K x 3) containing the indices, the predicted class
            and the `lc` of the k most informative samples
            column 1: indices
            column 2: predicted class.
            column 3: lc
        """
        assert np.round(pred_prob.sum(1).sum()) == pred_prob.shape[0], "pred_prob is not a probability distribution"
        assert 0 < k <= pred_prob.shape[0], "invalid k value k should be >0 & k <=  pred_prob.shape[0"
        # Get max probabilities prediction and its corresponding classes
        most_pred_prob, most_pred_class = np.max(pred_prob, axis=1), np.argmax(pred_prob, axis=1)
        size = len(pred_prob)
        lc_i = np.column_stack((list(range(size)), most_pred_class, most_pred_prob))
        # sort lc_i in ascending order
        lc_i = lc_i[lc_i[:, -1].argsort()]
        return lc_i[:k, 0].astype(np.int32), lc_i[:k]

    @staticmethod
    def margin_sampling(pred_prob, k):
        """
        Rank all the unlabeled samples in an ascending order according to the
        equation 3
        ----------
        pred_prob : np.ndarray
            prediction probability of x_i with dimension (batch x n_class)
        k : int
            most informative samples
    
        Returns
        -------
        np.array with dimension (K x 1)  containing the indices of the K
            most informative samples.
        np.array with dimension (K x 3) containing the indices, the predicted class
            and the `ms_i` of the k most informative samples
            column 1: indices
            column 2: predicted class.
            column 3: margin sampling
        """
        assert np.round(pred_prob.sum(1).sum()) == pred_prob.shape[0], "pred_prob is not a probability distribution"
        assert 0 < k <= pred_prob.shape[0], "invalid k value k should be >0 & k <=  pred_prob.shape[0"
        # Sort pred_prob to get j1 and j2
        size = len(pred_prob)
        margin = np.diff(np.abs(np.sort(pred_prob, axis=1)[:, ::-1][:, :2]))
        pred_class = np.argmax(pred_prob, axis=1)
        ms_i = np.column_stack((list(range(size)), pred_class, margin))

        # sort ms_i in ascending order according to margin
        ms_i = ms_i[ms_i[:, 2].argsort()]

        # the smaller the margin  means the classifier is more uncertain about the sample
        return ms_i[:k, 0].astype(np.int32), ms_i[:k]

    @staticmethod
    def entropy(pred_prob, k):
        """
        Rank all the unlabeled samples in an descending order according to
        the equation 4
    
        Parameters
        ----------
        pred_prob : np.ndarray
            prediction probability of x_i with dimension (batch x n_class)
        k : int
    
        Returns
        -------
        np.array with dimension (K x 1)  containing the indices of the K
            most informative samples.
        np.array with dimension (K x 3) containing the indices, the predicted class
            and the `en_i` of the k most informative samples
            column 1: indices
            column 2: predicted class.
            column 3: entropy
    
        """
        # calculate the entropy for the pred_prob
        assert np.round(pred_prob.sum(1).sum()) == pred_prob.shape[0], "pred_prob is not a probability distribution"
        assert 0 < k <= pred_prob.shape[0], "invalid k value k should be >0 & k <=  pred_prob.shape[0"
        size = len(pred_prob)
        entropy_ = - np.nansum(pred_prob * np.log(pred_prob), axis=1)
        pred_class = np.argmax(pred_prob, axis=1)
        en_i = np.column_stack((list(range(size)), pred_class, entropy_))

        # Sort en_i in descending order
        en_i = en_i[(-1 * en_i[:, 2]).argsort()]
        return en_i[:k, 0].astype(np.int32), en_i[:k]

    @classmethod
    def get_high_confidence_samples(cls, pred_prob, delta):
        """
        Select high confidence samples from `D^U` whose entropy is smaller than the threshold `delta`.
    
        Parameters
        ----------
        pred_prob : np.ndarray
            prediction probability of x_i with dimension (batch x n_class)
        delta : float
            threshold
    
        Returns
        -------
        np.array with dimension (K x 1)  containing the indices of the K most informative samples.
        np.array with dimension (K x 1) containing the predicted classes of the k most informative samples
        """
        _, eni = cls.entropy(pred_prob=pred_prob, k=len(pred_prob))
        hcs = eni[eni[:, 2] < delta]
        return hcs[:, 0].astype(np.int32), hcs[:, 1].astype(np.int32)

    @classmethod
    def get_uncertain_samples(cls, pred_prob, k, criteria):
        """
        Get the K most informative samples based on the criteria Parameters
        ----------
        pred_prob : np.ndarray
            prediction probability of x_i with dimension (batch x n_class)
        k: int
        criteria: str
            `cl` : least_confidence()
            `ms` : margin_sampling()
            `en` : entropy
    
        Returns
        -------
        tuple(np.ndarray, np.ndarray)
        """
        if criteria == 'cl':
            uncertain_samples = cls.least_confidence(pred_prob=pred_prob, k=k)
        elif criteria == 'ms':
            uncertain_samples = cls.margin_sampling(pred_prob=pred_prob, k=k)
        elif criteria == 'en':
            uncertain_samples = cls.entropy(pred_prob=pred_prob, k=k)
        else:
            raise ValueError('criteria {} not found !'.format(criteria))
        return uncertain_samples

    pass


class AlexNet(object):

    def __init__(self, n_classes=256, device=None):
        self.n_classes = n_classes
        self.model = alexnet(pretrained=True, progress=True)

        self.__freeze_all_layers()
        self.__change_last_layer()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass

    def __freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
            pass
        pass

    def __change_last_layer(self):
        self.model.classifier[6] = nn.Linear(4096, self.n_classes)
        pass

    def __add_softmax_layer(self):
        self.model = nn.Sequential(self.model, nn.LogSoftmax(dim=1))
        pass

    def __train_one_epoch(self, train_loader, optimizer, criterion, epoch, valid_loader=None, print_freq=100):
        train_loss = 0
        data_size = 0
        for batch_idx, sample_batched in enumerate(train_loader):
            # load data and label
            data, label = sample_batched['image'], sample_batched['label']
            data = data.to(self.device).float()
            label = label.to(self.device)

            optimizer.zero_grad()
            pred_prob = self.model(data)
            loss = criterion(pred_prob, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            data_size += label.size(0)

            if batch_idx % print_freq == print_freq - 1:
                Tools.print('Train Epoch: {} [{}/{}] Loss:{:.5f}'.format(
                    epoch, batch_idx, len(train_loader), loss.item()))
                pass
            pass
        if valid_loader:
            acc = self.evaluate(test_loader=valid_loader)
            Tools.print('Accuracy on the valid dataset {}'.format(acc))
            pass

        Tools.print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / data_size))
        pass

    def train(self, epochs, train_loader, valid_loader=None):
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001, momentum=0.9)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.__train_one_epoch(train_loader=train_loader, optimizer=optimizer,
                                   criterion=criterion, valid_loader=valid_loader, epoch=epoch)
            pass
        pass

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_loader):
                data, label = sample_batched['image'], sample_batched['label']
                data = data.to(self.device)
                data = data.float()
                label = label.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                pass
            pass
        return 100 * correct / total

    def predict(self, test_loader):
        self.model.eval()
        self.model.to(self.device)
        predict_results = np.empty(shape=(0, 256))
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_loader):
                data, _ = sample_batched['image'], sample_batched['label']
                data = data.to(self.device)
                data = data.float()
                outputs = self.model(data)
                outputs = softmax(outputs)
                predict_results = np.concatenate((predict_results, outputs.cpu().numpy()))
                pass
            pass
        return predict_results

    pass


def ceal_learning_algorithm(du, dl, d_test, k=1000, delta_0=0.005,
                            dr=0.00033, t=1, epochs=10, criteria='cl', max_iter=45):
    # Create the model
    model = AlexNet(n_classes=256)

    # Initialize the model
    Tools.print('Init dl={} du={} d_test={}'.format(
        len(dl.sampler.indices), len(du.sampler.indices), len(d_test.dataset)))
    model.train(epochs=epochs, train_loader=dl)
    acc = model.evaluate(test_loader=d_test)
    Tools.print('====> Initial accuracy: {} '.format(acc))

    for iteration in range(max_iter):
        Tools.print('Iteration: {}: run prediction on unlabeled data `du` '.format(iteration))
        pred_prob = model.predict(test_loader=du)

        # get k uncertain samples
        uncert_samp_idx, _ = Criteria.get_uncertain_samples(pred_prob=pred_prob, k=k, criteria=criteria)
        uncert_samp_idx = [du.sampler.indices[idx] for idx in uncert_samp_idx]
        dl.sampler.indices.extend(uncert_samp_idx)
        Tools.print('Update size of dl and du by adding uncertain {} samples. `dl` dl:{}, du:{}'.format(
            len(uncert_samp_idx), len(dl.sampler.indices),len(du.sampler.indices)))

        # get high confidence samples `dh`
        hcs_idx, hcs_labels = Criteria.get_high_confidence_samples(pred_prob=pred_prob, delta=delta_0)
        hcs_idx = [du.sampler.indices[idx] for idx in hcs_idx]
        hcs_idx = [x for x in hcs_idx if x not in list(set(uncert_samp_idx) & set(hcs_idx))]

        # add high confidence samples to the labeled set 'dl'
        dl.sampler.indices.extend(hcs_idx)  # (1) update the indices
        for idx in range(len(hcs_idx)):
            dl.dataset.label[hcs_idx[idx]] = hcs_labels[idx]  # (2) update the original labels with the pseudo labels.
        Tools.print('Update size of dl and du by adding {} hcs samples. dl:{}, du:{}'.format(
            len(hcs_idx), len(dl.sampler.indices), len(du.sampler.indices)))

        if iteration % t == 0:
            Tools.print('Iteration: {} fine-tune the model on dh U dl'.format(iteration))
            model.train(epochs=epochs, train_loader=dl)
            delta_0 = Criteria.update_threshold(delta=delta_0, dr=dr, t=iteration)
            pass

        Tools.print('remove {} uncertain samples from du'.format(len(uncert_samp_idx)))
        for val in uncert_samp_idx:
            du.sampler.indices.remove(val)

        acc = model.evaluate(test_loader=d_test)
        Tools.print("Iteration: {}, len(dl): {}, len(du): {}, len(dh) {}, acc: {} ".format(
            iteration, len(dl.sampler.indices), len(du.sampler.indices), len(hcs_idx), acc))
        pass

    pass


def main(data_root, batch_size=16, init_label_split=0.1):
    transform = transforms.Compose([SquarifyImage(), RandomCrop(224), Normalize(), ToTensor()])
    data_split = CalTech256Split(root_dir=data_root, split_test=0.2)
    dataset_train = CalTech256Dataset(transform=transform, data=data_split.data_train, label=data_split.label_train)
    dataset_test = CalTech256Dataset(transform=transform, data=data_split.data_test, label=data_split.label_test)

    indices = list(range(len(dataset_train)))
    np.random.shuffle(indices)
    init_label_num = int(init_label_split * len(dataset_train))

    loader_du = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(indices[init_label_num:]), num_workers=4)
    loader_dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(indices[:init_label_num]), num_workers=4)
    loader_d_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=4)

    ceal_learning_algorithm(du=loader_du, dl=loader_dl, d_test=loader_d_test)
    pass


class Config(object):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if "Linux" in platform.platform():
        batch_size = 16
        data_root = "/mnt/4T/Data/data/Caltech256/256_ObjectCategories"
    else:
        batch_size = 16
        data_root = "D:\\Data\\Caltech\\Caltech256\\256_ObjectCategories"
        pass

    pass


if __name__ == "__main__":
    main(data_root=Config.data_root, batch_size=Config.batch_size)
    pass
