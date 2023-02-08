import torch
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile


class VOC2007(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(VOC2007, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform

        self.images = list()
        self.labels = list()
        
            
        self.root_dir = root
        dict_path = root + 'VOCdevkit/VOC2007/ImageSets/Main/'
        categories = [file[:-10] for file in os.listdir(dict_path) if '_train.txt' in file]
        
        train_img = []
        test_img = []
        
        for file in os.listdir(dict_path):
            if '_train.txt' in file:
                fo = open(dict_path + file)
                cat = categories.index(file[:-10])
                l = [(line[:-4], cat) for line in iter(fo) if int(line[-3:]) == 1]
                train_img.extend(l)
            elif '_test.txt' in file:
                fo = open(dict_path + file)
                cat = categories.index(file[:-9])
                l = [(line[:-4], cat) for line in iter(fo) if int(line[-3:]) == 1]
                test_img.extend(l)
        
        train_img.sort()
        test_img.sort()
        self.train_img = train_img
        
        if self.train:
            image_set = train_img
        else:
            image_set = test_img
        for x in image_set:
            self.images.append(x[0])
            self.labels.append(x[1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
    
        image_path = os.path.join(self.root_dir, 'VOC2007', 'JPEGImages', self.images[idx]+'.jpg')
        img = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        target = torch.tensor(self.labels[idx])
        
        return img, target

