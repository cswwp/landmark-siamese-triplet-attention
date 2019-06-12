import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
import sys
from torchvision import transforms
import cv2

sys.path.append('/home/wangwenpeng/work/siamese-triplet-retriveal/data')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    #print('pillllllll:', path)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, label_to_indices, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(label_to_indices.keys())
        self.label_to_indices = label_to_indices
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

####################################################################
def read_txt(file_path, split_dem='\t'):
    content = open(file_path, 'r').readlines()
    im_lst = []
    labels = []

    for i in range(len(content)):
        lst = content[i].strip().split(split_dem)
        im_lst.append(lst[0])
        labels.append(lst[1])

    return im_lst, labels

def read_label2indices(la2indices_txt):
    rows = open(la2indices_txt, 'r').readlines()
    res = dict()
    for row in rows:
        lst = row.strip().split(' ')
        if lst[0] not in res.keys():
            res[lst[0]] = []

        for i in range(1, len(lst)):
            res[lst[0]].append(int(lst[i]))

    return res

class LandmarkDataset(Dataset):
    def __init__(self, root, mode, transform=None, transform_label=None, txt_base='/home/wangwenpeng/work/siamese-triplet-retriveal/data/stage2'):
        self.mode = mode
        self.traintxt = os.path.join(txt_base, 'train.txt')
        self.valtxt = os.path.join(txt_base, 'test.txt')
        self.tr_label2indices_txt = os.path.join(txt_base, 'tr_label2index.txt')
        self.te_label2indices_txt = os.path.join(txt_base, 'te_label2index.txt')
        self.transform = transform
        self.loader = pil_loader
        self.root = root
        self.transform_label = transform_label
        if self.mode == 'train':
            self.data, self.labels = read_txt(self.traintxt)
            self.label_to_indices = read_label2indices(self.tr_label2indices_txt)
        else:
            self.data, self.labels = read_txt(self.valtxt)
            self.label_to_indices = read_label2indices(self.te_label2indices_txt)

    def __getitem__(self, index):
        path = self.data[index] + '.jpg'
        target = int(self.labels[index])
        sample = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.data)

def init_data_loader(root, n_classes, n_samples, transform, transform_te, transform_label, kwargs):
    train_dataset = LandmarkDataset(root, 'train',
                                    transform, transform_label)
    test_dataset = LandmarkDataset(root, 'test',
                                   transform_te, transform_label)
    # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
    train_batch_sampler = BalancedBatchSampler(train_dataset.labels, train_dataset.label_to_indices,
                                               n_classes=n_classes, n_samples=n_samples)
    test_batch_sampler = BalancedBatchSampler(test_dataset.labels, test_dataset.label_to_indices, n_classes=n_classes,
                                              n_samples=n_samples)
    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
    return online_train_loader, online_test_loader

def init_transform(input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.3), ratio=(3. / 4., 4. / 3.)),
        transforms.RandomRotation(20),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.78, 0.78, 0.78),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=( 0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(input_size, scale=(0.7, 1.3), ratio=(3. / 4., 4. / 3.)),
        # transforms.RandomRotation(20),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.78, 0.78, 0.78),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])



    transform_label = transforms.Compose([transforms.ToTensor()])
    return transform, transform_test, transform_label

def inv_normalize(tensor):
    '''

    :param image_tensor:nchw
    :return: nchw
    '''
    image_tensor = tensor
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # res = np.zeros(image_tensor.shape)
    # print('fsafdsaf:', image_tensor)
    # print('31231:', image_tensor.size())
    # a = input()
    image_tensor[:, 0, :, :] = image_tensor[:, 0, :, :] * std[0] + mean[0]
    image_tensor[:, 1, :, :] = image_tensor[:, 1, :, :] * std[1] + mean[1]
    image_tensor[:, 2, :, :] = image_tensor[:, 2, :, :] * std[2] + mean[2]
    return image_tensor

def get_eval_txt(root='/data5/wwp/landmark', index_txt='index_data.txt', query_txt='query_data.txt'):
    index_path = os.path.join(root, 'index', 'train')
    query_path = os.path.join(root, 'test', 'test-full-size')
    lst_index = os.listdir(index_path)
    lst_query = os.listdir(query_path)

    w_index = open(index_txt, 'w')
    for item in lst_index:
        w_index.write(item + '\n')
    w_index.close()

    w_query = open(query_txt, 'w')
    for item in lst_query:
        w_query.write(item + '\n')
    w_query.close()

def read_eval_txt(path_txt):
    lst = open(path_txt, 'r').readlines()
    lst = [elem.strip() for elem in lst]
    return lst

class TestDataset(Dataset):
    def __init__(self, root='/data5/wwp/landmark', mode='index_data', transform=None):
        self.transform = transform
        self.loader = pil_loader
        self.root = root
        self.mode = mode
        if self.mode == 'index_data':
            self.root = os.path.join(self.root, 'index', 'train')
            self.data = read_eval_txt('/home/wangwenpeng/work/siamese-triplet-retriveal/data/index_data.txt')
        elif self.mode == 'query_data':
            self.root = os.path.join(self.root, 'test', 'test-full-size')
            self.data = read_eval_txt('/home/wangwenpeng/work/siamese-triplet-retriveal/data/query_data.txt')

    def __getitem__(self, index):
        fn = self.data[index]
        sample = self.loader(os.path.join(self.root, fn))
        # print('fsadfsda:', sample.size)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, fn.split('.')[0]

    def __len__(self):
        return len(self.data)


def TEST_DATA_LOADER(kwargs):
    mode = 'index_data'
    _, transform, _ = init_transform(224)
    index_data_set = TestDataset(mode=mode, transform=transform)
    mode = 'query_data'
    query_data_set = TestDataset(mode=mode, transform=transform)
    index_data_loader = torch.utils.data.DataLoader(index_data_set, shuffle=False, **kwargs)
    query_data_loader = torch.utils.data.DataLoader(query_data_set, shuffle=False, **kwargs)

    return index_data_loader, query_data_loader

if __name__ == '__main__':
    cuda = 1
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.485, 0.485), std=(0.229, 0.229, 0.229))])

    train_dataset = LandmarkDataset('/data5/wwp/landmark/train/Landmark-classification-resized-256/train/', 'train', transform)
    train_batch_sampler = BalancedBatchSampler(train_dataset.labels, train_dataset.label_to_indices, n_classes=1, n_samples=1)
    tr_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler)

    for data, label in tr_data_loader:
        print(data.shape)
        print(label)
        print('label2indices:', label[0], train_dataset.label_to_indices[label[0]])




    # cuda = 1
    # index_data_loader, query_data_loader = TEST_DATA_LOADER(kwargs = {'num_workers': 16, 'pin_memory': False} if cuda else {})
    #
    # for index_img, index_fn in index_data_loader:
    #     print(index_img)
    #     print(index_fn)
    #     a = input()