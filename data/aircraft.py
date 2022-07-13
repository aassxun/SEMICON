package_paths = [
    "/home/ubuntu/Project/test/fghash/data",
    "/home/ubuntu/Project/test/fghash"
]
import sys;
for pth in package_paths:
    sys.path.append(pth)
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
    Aircraft.init(root)
    query_dataset = Aircraft(root, 'query', query_transform())
    train_dataset = Aircraft(root, 'train', train_transform())
    retrieval_dataset = Aircraft(root, 'retrieval', query_transform())
    print(len(query_dataset))
    print(len(train_dataset))

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


class Aircraft(Dataset):

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = Aircraft.TRAIN_DATA
            self.targets = Aircraft.TRAIN_TARGETS
        elif mode == 'query':
            self.data = Aircraft.QUERY_DATA
            self.targets = Aircraft.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Aircraft.RETRIEVAL_DATA
            self.targets = Aircraft.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')


    @staticmethod
    def init(root):
        image_class_labels = pd.read_csv(os.path.join(root, 'fgvc-aircraft-2013b/data/variants.txt'), names=['target'])
        d = {}
        for i in range(len(image_class_labels)):
            d[image_class_labels['target'][i]] = i+1

        images_train = pd.read_csv(os.path.join(root,'fgvc-aircraft-2013b/data/images_variant_trainval.txt'), names=['filepath'])
        images_test = pd.read_csv(os.path.join(root,'fgvc-aircraft-2013b/data/images_variant_test.txt'), names=['filepath'])
        train_images = []
        test_images = []
        for i in range(len(images_train)):
            # print(images_train['filepath'][i])
            train_images.append(images_train['filepath'][i].split(' ')[0] + '.jpg')
        for i in range(len(images_test)):
            test_images.append(images_test['filepath'][i].split(" ")[0] + '.jpg')
        # print(images_train[:10])
        label_list_train = []
        img_id_train = []
        for i in range(len(train_images)):
            # print(images_train['filepath'][i])
            label = images_train['filepath'][i].split(' ',1)[1]
            label_list_train.append(d[label])
            img_id_train.append(i+1)
        # print(label_list_train[:10])
        images_train = []
        for i in range(len(train_images)):
            images_train.append([img_id_train[i], 'fgvc-aircraft-2013b/data/images/'+train_images[i], label_list_train[i]])

        images_train = pd.DataFrame(images_train, columns=['img_id', 'filepath', 'target'])
        k = len(train_images)
        label_list_test = []
        img_id_test = []
        for i in range(len(test_images)):
            label = images_test['filepath'][i].split(' ',1)[1]
            label_list_test.append(d[label])
            img_id_test.append(k+i+1)
        images_test = []
        for i in range(len(test_images)):
            images_test.append([img_id_test[i], 'fgvc-aircraft-2013b/data/images/'+test_images[i], label_list_test[i]])
        images_test = pd.DataFrame(images_test, columns=['img_id', 'filepath', 'target'])

        train_data = images_train
        test_data = images_test


        # Split dataset
        Aircraft.QUERY_DATA = test_data['filepath'].to_numpy()
        Aircraft.QUERY_TARGETS = encode_onehot((test_data['target'] - 1).tolist(), 100)

        Aircraft.TRAIN_DATA = train_data['filepath'].to_numpy()
        Aircraft.TRAIN_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 100)

        Aircraft.RETRIEVAL_DATA = train_data['filepath'].to_numpy()
        Aircraft.RETRIEVAL_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 100)


    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx], idx


# def main():
#     query_dataloader, train_dataloader, retrieval_dataloader=load_data('/dataset/aircraft/', 16, 4)

# main()