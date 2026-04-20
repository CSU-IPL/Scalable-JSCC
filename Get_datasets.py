import os
from PIL import Image
from torch.utils.data import Dataset

class OneImageDataset(Dataset):
    def __init__(self, dir, transform):
        self.dir = dir
        self.images = sorted(os.listdir(dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.images[idx])
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


class PairedImageDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.low_res_images = sorted(os.listdir(low_res_dir))
        self.high_res_images = sorted(os.listdir(high_res_dir))
        self.transform = transform

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res_path = os.path.join(self.low_res_dir, self.low_res_images[idx])
        high_res_path = os.path.join(self.high_res_dir, self.high_res_images[idx])

        low_res_img = Image.open(low_res_path).convert("RGB")
        high_res_img = Image.open(high_res_path).convert("RGB")

        if self.transform:
            low_res_img = self.transform(low_res_img)
            high_res_img = self.transform(high_res_img)

        return low_res_img, high_res_img


class MultiResolutionDataset(Dataset):
    def __init__(self, low_res_dir, medium_res_dir, high_res_dir, transform=None):
        self.low_res_images = sorted(os.listdir(low_res_dir))
        self.medium_res_images = sorted(os.listdir(medium_res_dir))
        self.high_res_images = sorted(os.listdir(high_res_dir))
        self.low_res_dir = low_res_dir
        self.medium_res_dir = medium_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform

    def __len__(self):
        return len(self.low_res_images)  # Assuming the same number of images in each folder

    def __getitem__(self, idx):
        low_res_image = Image.open(os.path.join(self.low_res_dir, self.low_res_images[idx]))
        medium_res_image = Image.open(os.path.join(self.medium_res_dir, self.medium_res_images[idx]))
        high_res_image = Image.open(os.path.join(self.high_res_dir, self.high_res_images[idx]))

        if self.transform:
            low_res_image = self.transform(low_res_image)
            medium_res_image = self.transform(medium_res_image)
            high_res_image = self.transform(high_res_image)

        return low_res_image, medium_res_image, high_res_image
