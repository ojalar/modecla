from torch.utils.data.dataset import Dataset
import torchvision.transforms
import pandas as pd
import numpy as np
from PIL import Image
import custom_transforms

# This is the implementation of the custom classification dataset
# Image paths are expected in a csv file in format:
# <image path>, <class>
class ClassificationDataset(Dataset):
    def __init__(self, path, transforms, train):
        self.data_info = pd.read_csv(path, header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.data_len = len(self.data_info.index)
        self.transforms = transforms
        self.train = train

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name).convert('RGB')
        label = self.label_arr[index]
        x, y = img_as_img.size
        # training transformations applied before black bars conversion
        if self.train:
            train_transform = custom_transforms.RandomBlur(0.8, (1, 6))
            img_as_img = train_transform(img_as_img)
            train_transform = torchvision.transforms.RandomCrop((y,x), padding=(int(x/8),int(y/8)))
            img_as_img = train_transform(img_as_img)

        size = max((x,y))
        new_img_as_img = Image.new('RGB', (size, size))
        new_img_as_img.paste(img_as_img, (int(((size - x) / 2)), int((size - y) / 2)))
        img = self.transforms(new_img_as_img)
        return (img, label)

    def __len__(self):
        return self.data_len
