import torchvision.transforms as transforms

import os

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image


def load_data(root, batch_size, workers):
    """
    Load imagenet dataset

    Args:
        root (str): Path of imagenet dataset.
        batch_size (int): Number of samples in one batch.
        workers (int): Number of data loading threads.

    Returns:
        train_loader (torch.utils.data.DataLoader): Training dataset loader.
        query_loader (torch.utils.data.DataLoader): Query dataset loader.
        val_loader (torch.utils.data.DataLoader): Validation dataset loader.
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    query_val_init_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # Construct data loader
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')

    train_dataset = ImagenetDataset(
        traindir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    query_dataset = ImagenetDataset(
        valdir,
        transform=query_val_init_transform,
        num_samples=5000,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )

    val_dataset = ImagenetDataset(
        valdir,
        transform=query_val_init_transform,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return query_loader, train_loader, val_loader


class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, transform=None, num_samples=None):
        self.root = root
        self.transform = transform
        self.imgs = []
        self.targets = []

        # Assume file alphabet order is the class order
        if ImagenetDataset.class_to_idx is None:
            ImagenetDataset.classes, ImagenetDataset.class_to_idx = self._find_classes(root)

        for i, cl in enumerate(ImagenetDataset.classes):
            cur_class = os.path.join(self.root, cl)
            files = os.listdir(cur_class)
            if num_samples is not None:
                num_per_class = num_samples // len(ImagenetDataset.classes)
                sample_files = files[: num_per_class]
                sample_files = [os.path.join(cur_class, file) for file in sample_files]

                self.imgs.extend(sample_files)
                self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(num_per_class)])
            else:
                files = [os.path.join(cur_class, i) for i in files]
                self.imgs.extend(files)
                self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(len(files))])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img, target = self.imgs[item], self.targets[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, target, item

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
