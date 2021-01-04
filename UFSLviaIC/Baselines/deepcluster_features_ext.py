import os
import math
import torch
import platform
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class VGG(nn.Module):

    def __init__(self, features, num_classes, sobel):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )
        self.top_layer = nn.Linear(4096, num_classes)
        self._initialize_weights()
        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0,0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1,0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.classifier:
            x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                #print(y)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    pass


def make_layers(input_dim, batch_norm):
    layers = []
    in_channels = input_dim
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(sobel=False, bn=True, out=1000):
    dim = 2 + int(not sobel)
    model = VGG(make_layers(dim, bn), out, sobel)
    return model


class MyDataset(Dataset):

    def __init__(self, data_list, image_size=84):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.CenterCrop(size=image_size), transforms.ToTensor(), normalize])
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image_transform = self.transform(image)
        return image_transform, label, idx

    @staticmethod
    def get_data_all(data_root):
        train_folder = os.path.join(data_root, "train")
        val_folder = os.path.join(data_root, "val")
        test_folder = os.path.join(data_root, "test")

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

        count_image, count_class, data_val_list = 0, 0, []
        for label in os.listdir(val_folder):
            now_class_path = os.path.join(val_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_val_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        count_image, count_class, data_test_list = 0, 0, []
        for label in os.listdir(test_folder):
            now_class_path = os.path.join(test_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_test_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        return data_train_list, data_val_list, data_test_list

    pass


class ExtFeatures(object):

    def __init__(self, checkpoint_path="./checkpoint/vgg16/checkpoint.pth.tar"):
        self.model = self.load_model(checkpoint_path).cuda()
        self.model.top_layer = None
        self.model.classifier = None
        pass

    @staticmethod
    def load_model(path):
        """Loads model and return it without DataParallel table."""
        if os.path.isfile(path):
            Tools.print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)

            # size of the top layer
            n = checkpoint['state_dict']['top_layer.bias'].size()

            # build skeleton of the model
            sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
            model = vgg16(sobel=sob, out=int(n[0]))

            # deal with a dataparallel table
            def rename_key(key):
                if not 'module' in key:
                    return key
                return ''.join(key.split('.module'))

            checkpoint['state_dict'] = {rename_key(key): val for key, val in checkpoint['state_dict'].items()}

            # load weights
            model.load_state_dict(checkpoint['state_dict'])
            Tools.print("Loaded")
        else:
            Tools.print("=> no checkpoint found at '{}'".format(path))
            raise Exception("....")
        return model

    def run_features(self, data_list):
        output_feature = []
        with torch.no_grad():
            self.model.eval()

            loader = DataLoader(MyDataset(data_list), Config.batch_size, False, num_workers=Config.num_workers)
            for image, label, idx in tqdm(loader):
                image = image.cuda()
                output = self.model(image).data.cpu().numpy()
                output_feature.extend(output)
                pass
            pass
        return output_feature

    pass


class Config(object):
    gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_workers = 8
    batch_size = 8

    is_png = True

    # dataset_name = "miniimagenet"
    dataset_name = "tieredimagenet"

    if dataset_name == "miniimagenet":
        if "Linux" in platform.platform():
            data_root = '/mnt/4T/Data/data/miniImagenet'
            if not os.path.isdir(data_root):
                data_root = '/media/ubuntu/4T/ALISURE/Data/miniImagenet'
        else:
            data_root = "F:\\data\\miniImagenet"
        data_root = os.path.join(data_root, "miniImageNet_png") if is_png else data_root
    else:
        if "Linux" in platform.platform():
            data_root = '/mnt/4T/Data/data/UFSL/tiered-imagenet'
            if not os.path.isdir(data_root):
                data_root = '/media/ubuntu/4T/ALISURE/Data/UFSL/tiered-imagenet'
        else:
            data_root = "F:\\data\\UFSL\\tiered-imagenet"
    Tools.print(data_root)

    features_save_path = Tools.new_dir("{}_feature".format(data_root))
    Tools.print(features_save_path)
    pass


if __name__ == '__main__':
    data_train, data_val, data_test = MyDataset.get_data_all(Config.data_root)

    ext_features = ExtFeatures()

    features_train = ext_features.run_features(data_list=data_train)
    Tools.write_to_pkl(os.path.join(Config.features_save_path, "train_features.pkl"),
                       {"info": data_train, "feature": features_train})

    features_val = ext_features.run_features(data_list=data_val)
    Tools.write_to_pkl(os.path.join(Config.features_save_path, "val_features.pkl"),
                       {"info": data_val, "feature": features_val})

    features_test = ext_features.run_features(data_list=data_test)
    Tools.write_to_pkl(os.path.join(Config.features_save_path, "test_features.pkl"),
                       {"info": data_test, "feature": features_test})
    pass
