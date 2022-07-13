import warnings
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, extract_archive

import torch
import numpy as np

import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image, ImageFile

from data.transform import encode_onehot
from data.transform import train_transform, query_transform

def load_data(root, batch_size, num_workers):
    NABirds.init(root)
    query_dataset = NABirds(root, 'query', query_transform())
    train_dataset = NABirds(root, 'train', train_transform())
    retrieval_dataset = NABirds(root, 'retrieval', query_transform())
    print(len(query_dataset))
    print(len(train_dataset))
    print(len(retrieval_dataset))
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader




class NABirds(VisionDataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'images/'
    filename = 'nabirds.tar.gz'
    md5 = 'df21a9e4db349a14e2b08adfd45873bd'

    def __init__(self, root, mode, transform=None, target_transform=None):
        super(NABirds, self).__init__(root, transform=transform, target_transform=target_transform)

        dataset_path = root
        if not os.path.isdir(dataset_path):
            if not check_integrity(os.path.join(root, self.filename), self.md5):
                raise RuntimeError('Dataset not found or corrupted.')
            extract_archive(os.path.join(root, self.filename))
        self.root = os.path.expanduser(root)
        self.loader = default_loader
        # self.transform = transform
                # Load in the class data
        self.class_names = load_class_names(root)
        self.class_hierarchy = load_hierarchy(root)

        if mode == 'train':
            self.data = NABirds.TRAIN_DATA
            self.targets = NABirds.TRAIN_TARGETS
        elif mode == 'query':
            self.data = NABirds.QUERY_DATA
            self.targets = NABirds.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = NABirds.RETRIEVAL_DATA
            self.targets = NABirds.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')


    @staticmethod
    def init(root):
        image_paths = pd.read_csv(os.path.join(root, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        all_data = data.merge(train_test_split, on='img_id')
        all_data['filepath'] = 'images/' + all_data['filepath']
        all_data['target'] = all_data['target'].apply(lambda x: label_map[x])

        train_data = all_data[all_data['is_training_img'] == 1]
        # test_data = all_data[all_data['is_training_img'] == 0].iloc[5001:10551, :]
        test_data = all_data[all_data['is_training_img'] == 0]
        class_num = len(label_map)
        # Load in the train / test split
        NABirds.QUERY_DATA = test_data['filepath'].to_numpy()
        NABirds.QUERY_TARGETS = encode_onehot((test_data['target']).tolist(), class_num)
        NABirds.TRAIN_DATA = train_data['filepath'].to_numpy()
        NABirds.TRAIN_TARGETS = encode_onehot((train_data['target']).tolist(), class_num)
        NABirds.RETRIEVAL_DATA = train_data['filepath'].to_numpy()
        NABirds.RETRIEVAL_TARGETS = encode_onehot((train_data['target']).tolist(), class_num)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx], idx


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents


# if __name__ == '__main__':
#     train_dataset = NABirds('./nabirds', train=True, download=False)
#     test_dataset = NABirds('./nabirds', train=False, download=False)
