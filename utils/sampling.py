#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users, imbalance_degree=0.1):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    #num_imgs: how many images each client has
    # num_shards
    num_shards, num_imgs = 60, 1000

    imbalance_num = int(num_imgs * imbalance_degree)
    other_num = int((1 - imbalance_degree) / 9 * num_imgs)
    print("imbalance number:",imbalance_num," and other number:",other_num)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)} #以字典形式返回给用户的数据
    idxs = np.arange(num_shards*num_imgs) # 获得所有图片的索引
    labels = dataset.train_labels.numpy()  # 获得所有图片的标签

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    idx_list = []
    idx_list.append(idxs[0:5922])
    idx_list.append(idxs[5923:12664])
    idx_list.append(idxs[12665:18622])
    idx_list.append(idxs[18623:24753])
    idx_list.append(idxs[24754:30595])
    idx_list.append(idxs[30596:36016])
    idx_list.append(idxs[36017:41934])
    idx_list.append(idxs[41935:48199])
    idx_list.append(idxs[48200:54050])
    idx_list.append(idxs[54051:])


    # divide and assign
    for i in range(num_users):
        '''
        这里non-iid的方法是随机选择两个标签，组成rand_set
        然后让idx_shard干掉rand_set里的标签，变成non-iid
        '''
        imbalance_number = np.random.choice([0,1,2,3,4,5,6,7,8,9])
        rand_set = set([imbalance_number])
        idx_shard = list(set([0,1,2,3,4,5,6,7,8,9]) - rand_set)
        client_idx = []
        for id in idx_shard:
            client_idx += list(np.random.choice(idx_list[id], other_num, replace=False))
        client_idx += list(np.random.choice(idx_list[imbalance_number], imbalance_num, replace=False))
        dict_users[i] = np.concatenate((dict_users[i], client_idx), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
