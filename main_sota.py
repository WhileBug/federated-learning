#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ResNetMnist
from models.Fed import FedAvg
from models.test import test_img, test_img_2
import pandas as pd
from sota.sota import Median as Median, TrimmedMean

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def FedAvg_test(untargeted_rate, random_rate, imbalance_degree, args_dataset, args_model, args, aggre_algo):
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    test_accuracy_list = []

    # load dataset and split users
    if args_dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            print("choose iid")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("choose non iid")
            dict_users = mnist_noniid(dataset_train, args.num_users, imbalance_degree)
    elif args_dataset == 'fashion':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/fashion/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('../data/fashion/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            print("choose iid")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("choose non iid")
            dict_users = mnist_noniid(dataset_train, args.num_users, imbalance_degree)
    elif args_dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args_model == 'cnn' and args_dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args_model == 'cnn' and args_dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args_model == 'resnet' and args_dataset == 'mnist':
        net_glob = ResNetMnist(args=args).to(args.device)
    elif args_model == 'resnet' and args_dataset == 'fashion':
        net_glob = ResNetMnist(args=args).to(args.device)
    elif args_model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        print(m)
        untargeted_num = int(untargeted_rate * m)
        random_num = int(random_rate * m)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:

            if(untargeted_num>0):
                #print("attack")
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], isUntargeted=True)
                untargeted_num -= 1
            elif(random_num>0):
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], isRandom=True)
                random_num -= 1
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # 根据对framework的选择，调用Median、Trimmed Mean或者FedAvg聚合函数，聚合local updates成global updates
        if(aggre_algo=="Median"):
            w_glob = Median(w_locals)
        elif(aggre_algo=="TrimmedMean"):
            w_glob = TrimmedMean(w_locals)
        elif(aggre_algo=="FedAvg"):
            w_glob = FedAvg(w_locals)
        # 将global updates后的weight复制给net_glob，即全局网络模型
        net_glob.load_state_dict(w_glob)

        # testing 增加在每一轮对global model的test
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Round {:3d},Training accuracy: {:.2f}".format(iter, acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        
        test_accuracy_list.append(acc_test)
    return test_accuracy_list

def exp_main(folder_addr, attack_rate_list, imbalance_degree_list, attack_type="random", dataset_type='mnist', model_type = 'resnet', framework_type = "Median"):
    args = args_parser()

    if(attack_type == 'untargeted'):
        random_rate = 0.0
        for untargeted_rate in attack_rate_list:
            temp_accuracy_list = []
            for imbalance_degree in imbalance_degree_list:
                accuracy_list = FedAvg_test(untargeted_rate, random_rate, imbalance_degree, dataset_type, model_type, args, framework_type)
                temp_accuracy_list.append(accuracy_list)
            temp_accuracy_dataframe = pd.DataFrame(temp_accuracy_list)
            temp_accuracy_dataframe.to_csv(folder_addr+"/"+framework_type+"/"+attack_type+"/"+dataset_type+"/"+str(untargeted_rate)+".csv")

    if (attack_type == 'random'):
        untargeted_rate_rate = 0.0
        for random_rate in attack_rate_list:
            temp_accuracy_list = []
            for imbalance_degree in imbalance_degree_list:
                accuracy_list = FedAvg_test(untargeted_rate, random_rate, imbalance_degree, dataset_type, model_type,
                                            args, framework_type)
                temp_accuracy_list.append(accuracy_list)
            temp_accuracy_dataframe = pd.DataFrame(temp_accuracy_list)
            temp_accuracy_dataframe.to_csv(
                folder_addr + "/" + framework_type + "/" + attack_type + "/" + dataset_type + "/" + str(
                    random_rate) + ".csv")



if __name__ == '__main__':
    folder_addr = "exp_data"
    attack_rate_list = [0.1,0.2,0.4,0.5]
    imbalance_degree_list = [0.1,0.2,0.4,0.5]
    attack_type = 'untargeted'
    dataset_type = 'mnist'
    model_type = 'resnet'
    framework_type = "Median"
    exp_main(folder_addr, attack_rate_list, imbalance_degree_list, attack_type, dataset_type, model_type, framework_type)