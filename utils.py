import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
from torchvision import datasets
# from model import *
from dataset import *
# from sampling import *
import ipdb
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())

    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))  


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_base_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform, split='test_base')
    cifar10_test_new_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform, split='test_new')
    
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
    X_test_base, y_test_base = cifar10_test_base_ds.data, cifar10_test_base_ds.targets
    X_test_new, y_test_new = cifar10_test_new_ds.data, cifar10_test_new_ds.targets

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test_base, y_test_base, X_test_new, y_test_new)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_base_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform, split='test_base')
    cifar100_test_new_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform, split='test_new')
    
    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.targets
    X_test_base, y_test_base = cifar100_test_base_ds.data, cifar100_test_base_ds.targets
    X_test_new, y_test_new = cifar100_test_new_ds.data, cifar100_test_new_ds.targets

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test_base, y_test_base, X_test_new, y_test_new)




def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    
    xray_train_ds = TinyImageNet_load('../datasets/tiny-imagenet-200/', train=True, transform=transform)
    xray_test_base_ds = TinyImageNet_load('../datasets/tiny-imagenet-200/', train=False, transform=transform, split='test_base')
    xray_test_new_ds = TinyImageNet_load('../datasets/tiny-imagenet-200/', train=False, transform=transform, split='test_new')

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_base_test, y_base_test = np.array([s[0] for s in xray_test_base_ds.samples]), np.array([int(s[1]) for s in xray_test_base_ds.samples])
    X_new_test, y_new_test = np.array([s[0] for s in xray_test_new_ds.samples]), np.array([int(s[1]) for s in xray_test_new_ds.samples])

    return (X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test)

def load_vireo_data():
    transform = transforms.Compose([transforms.ToTensor()])

    vireo_train_ds = Vireo172_truncated(transform=transform, mode='train')
    vireo_test_base_ds = Vireo172_truncated(transform=transform, mode='test_base')
    vireo_test_new_ds = Vireo172_truncated(transform=transform, mode='test_new')
    
    X_train, y_train = vireo_train_ds.path_to_images, vireo_train_ds.labels
    X_test_base, y_test_base = vireo_test_base_ds.path_to_images, vireo_test_base_ds.labels
    X_test_new, y_test_new = vireo_test_new_ds.path_to_images, vireo_test_new_ds.labels
    
    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test_base, y_test_base, X_test_new, y_test_new)


def load_food_data():
    transform = transforms.Compose([transforms.ToTensor()])

    food_train_ds = Food101_truncated(transform=transform, mode='train')
    food_test_base_ds = Food101_truncated(transform=transform, mode='test_base')
    food_test_new_ds = Food101_truncated(transform=transform, mode='test_new')

    X_train, y_train = food_train_ds.path_to_images, food_train_ds.labels
    X_test_base, y_test_base = food_test_base_ds.path_to_images, food_test_base_ds.labels
    X_test_new, y_test_new = food_test_new_ds.path_to_images, food_test_new_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test_base, y_test_base, X_test_new, y_test_new)

def load_cal101_data():
    transform = transforms.Compose([transforms.ToTensor()])

    cal101_train_ds = Caltech101(transform=transform, mode='train')
    cal101_test_base_ds = Caltech101(transform=transform, mode='test_base')
    cal101_test_new_ds = Caltech101(transform=transform, mode='test_new')

    X_train, y_train = cal101_train_ds.path_to_images, cal101_train_ds.labels
    X_test_base, y_test_base = cal101_test_base_ds.path_to_images, cal101_test_base_ds.labels
    X_test_new, y_test_new = cal101_test_new_ds.path_to_images, cal101_test_new_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test_base, y_test_base, X_test_new, y_test_new)

def load_flowers_data():
    transform = transforms.Compose([transforms.ToTensor()])

    flowers_train_ds = Flowers102(transform=transform, mode='train')
    flowers_test_base_ds = Flowers102(transform=transform, mode='test_base')
    flowers_test_new_ds = Flowers102(transform=transform, mode='test_new')
    X_train, y_train = flowers_train_ds.path_to_images, flowers_train_ds.labels
    X_base_test, y_base_test = flowers_test_base_ds.path_to_images, flowers_test_base_ds.labels
    X_new_test, y_new_test = flowers_test_new_ds.path_to_images, flowers_test_new_ds.labels
    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test)

def load_aircraft_data():
    transform = transforms.Compose([transforms.ToTensor()])

    air_train_ds = Air100(transform=transform, mode='train')
    air_test_base_ds = Air100(transform=transform, mode='test_base')
    air_test_new_ds = Air100(transform=transform, mode='test_new')

    X_train, y_train = air_train_ds.path_to_images, air_train_ds.labels
    X_base_test, y_base_test = air_test_base_ds.path_to_images, air_test_base_ds.labels
    X_new_test, y_new_test = air_test_new_ds.path_to_images, air_test_new_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test)


def load_cars_data():
    transform = transforms.Compose([transforms.ToTensor()])

    cars_train_ds = Cars196(transform=transform, mode='train')
    cars_test_base_ds = Cars196(transform=transform, mode='test_base')
    cars_test_new_ds = Cars196(transform=transform, mode='test_new')
    
    X_train, y_train = cars_train_ds.path_to_images, cars_train_ds.labels
    X_base_test, y_base_test = cars_test_base_ds.path_to_images, cars_test_base_ds.labels
    X_new_test, y_new_test = cars_test_new_ds.path_to_images, cars_test_new_ds.labels
    
    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test)


def load_ucf_data():
    transform = transforms.Compose([transforms.ToTensor()])

    ucf_train_ds = UCF101(transform=transform, mode='train')
    ucf_test_base_ds = UCF101(transform=transform, mode='test_base')
    ucf_test_new_ds = UCF101(transform=transform, mode='test_new')

    X_train, y_train = ucf_train_ds.path_to_images, ucf_train_ds.labels
    X_base_test, y_base_test = ucf_test_base_ds.path_to_images, ucf_test_base_ds.labels
    X_new_test, y_new_test = ucf_test_new_ds.path_to_images, ucf_test_new_ds.labels
    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test)

def load_dtd_data():
    transform = transforms.Compose([transforms.ToTensor()])

    dtd_train_ds = DTD(transform=transform, mode='train')
    dtd_test_base_ds = DTD(transform=transform, mode='test_base')
    dtd_test_new_ds = DTD(transform=transform, mode='test_new')

    X_train, y_train = dtd_train_ds.path_to_images, dtd_train_ds.labels
    X_base_test, y_base_test = dtd_test_base_ds.path_to_images, dtd_test_base_ds.labels
    X_new_test, y_new_test = dtd_test_new_ds.path_to_images, dtd_test_new_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test)

def load_pets_data():
    transform = transforms.Compose([transforms.ToTensor()])

    dtd_train_ds = Pets(transform=transform, mode='train')
    dtd_test_base_ds = Pets(transform=transform, mode='test_base')
    dtd_test_new_ds = Pets(transform=transform, mode='test_new')

    X_train, y_train = dtd_train_ds.path_to_images, dtd_train_ds.labels
    X_base_test, y_base_test = dtd_test_base_ds.path_to_images, dtd_test_base_ds.labels
    X_new_test, y_new_test = dtd_test_new_ds.path_to_images, dtd_test_new_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test)


def load_domain_data(dataset):
    X_train, y_train = np.array(dataset.data), np.array(dataset.labels)
    X_base_test, y_base_test, X_new_test, y_new_test = X_train, y_train, X_train, y_train

    return (X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, name_classes, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_tinyimagenet_data(datadir)
    elif dataset == 'vireo172':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_vireo_data()
    elif dataset == 'food101':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_food_data()          
    elif dataset == 'cal101':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_cal101_data()   
    elif dataset == 'flowers102':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_flowers_data()     
    elif dataset == 'air100':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_aircraft_data() 
    elif dataset == 'cars196':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_cars_data() 
    elif dataset == 'ucf101':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_ucf_data()        
    elif dataset == 'dtd':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_dtd_data() 
    elif dataset == 'pets':
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_pets_data() 
    elif "domain" in dataset:
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test = load_domain_data(datadir) 
        
        
    n_train = len(y_train)  
    net_classname_map = {}
    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    
    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100' or dataset == 'air100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset == 'vireo172':
            K = 172
        elif dataset == 'food101' or dataset == 'cal101' or dataset == 'ucf101':
            K = 101  
        elif dataset == 'flowers102':
            K = 102  
        elif dataset == 'cars196':
            K = 196     
        elif dataset == 'dtd':
            K = 47    
        elif dataset == 'pets':
            K = 37    
        elif dataset == "domain_office":
            K = 65
        elif dataset == "domain_vlcs":
            K = 5
            
        N = len(y_train)
        net_dataidx_map = {}
        # ipdb.set_trace()
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break
            print(min_size)
        net_unq_map = {}
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            unq, unq_cnt = np.unique(y_train[net_dataidx_map[j]], return_counts=True)
#             ipdb.set_trace()
            net_classname_map[j] = np.array(name_classes)[unq.astype(np.int)]
#             net_unq_map[j] = unq

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test, net_dataidx_map, net_classname_map, traindata_cls_counts)



def get_data(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/', train=False, download=True, transform=trans_mnist)

        X_train, y_train = dataset_train.data, np.array(dataset_train.targets)
#         X_train, y_train = dataset_train.data, torch.tensor(dataset_train.targets)
        X_test, y_test = dataset_test.data, np.array(dataset_test.targets)

        # sample users
        if args.partition == 'iid':
            dict_users_train = iid(dataset_train, args.n_parties)
            dict_users_test = iid(dataset_test, args.n_parties)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, y_train, args.n_parties, args.shard_per_user)
#             print('dict_users_train', dict_users_train)
#             dict_users_test, rand_set_all = noniid(dataset_test, args.n_parties, args.shard_per_user, rand_set_all=rand_set_all)
    elif args.dataset == 'cifar10':
        trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
        trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        dataset_train = datasets.CIFAR10('data/', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/', train=False, download=True, transform=trans_cifar10_val)
        
        X_train, y_train = dataset_train.data, np.array(dataset_train.targets)
#         X_train, y_train = dataset_train.data, torch.tensor(dataset_train.targets)
        X_test, y_test = dataset_test.data, np.array(dataset_test.targets)
        
        if args.partition == 'iid':
            dict_users_train = iid(dataset_train, args.n_parties)
            dict_users_test = iid(dataset_test, args.n_parties)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, y_train, args.n_parties, args.shard_per_user)
#             dict_users_test, rand_set_all = noniid(dataset_test, y_test, args.n_parties, args.shard_per_user, rand_set_all=rand_set_all)
    elif args.dataset == 'cifar100':
        trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
        trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        
        X_train, y_train = dataset_train.data, np.array(dataset_train.targets)
#         X_train, y_train = dataset_train.data, torch.tensor(dataset_train.targets)
        X_test, y_test = dataset_test.data, np.array(dataset_test.targets)
        
        if args.partition == 'iid':
            dict_users_train = iid(dataset_train, args.n_parties)
            dict_users_test = iid(dataset_test, args.n_parties)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, y_train, args.n_parties, args.shard_per_user)
#             dict_users_test, rand_set_all = noniid(dataset_test, args.n_parties, args.shard_per_user, rand_set_all=rand_set_all)
    elif args.dataset == 'fmnist':
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('data/', train=False, download=True, transform=trans_mnist)

        X_train, y_train = dataset_train.train_data, dataset_train.train_labels
        print(y_train)
        X_test, y_test = dataset_test.test_data, dataset_test.test_labels

        # sample users
        if args.partition == 'iid':
            dict_users_train = iid(dataset_train, args.n_parties)
            dict_users_test = iid(dataset_test, args.n_parties)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.n_parties, args.shard_per_user)
#             print('dict_users_train', dict_users_train)
            dict_users_test, rand_set_all = noniid(dataset_test, args.n_parties, args.shard_per_user, rand_set_all=rand_set_all)
   
    
    elif args.dataset == 'tinyimagenet':
        dl_obj = TinyImageNet_load
        transform_train = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset_train = dl_obj('data/tiny-imagenet-200/', transform=transform_train)
        dataset_test = dl_obj('data/tiny-imagenet-200/', transform=transform_test)

        X_train, y_train = np.array(dataset_train.samples)[:][0], np.array(dataset_train.samples)[:,1]
#         X_train, y_train = dataset_train.data, torch.tensor(dataset_train.targets)
        X_test, y_test = np.array(dataset_test.samples)[:][0], np.array(dataset_test.samples)[:,1]
        
        if args.partition == 'iid':
            dict_users_train = iid(dataset_train, args.n_parties)
            dict_users_test = iid(dataset_test, args.n_parties)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, y_train, args.n_parties, args.shard_per_user)
#             dict_users_test, rand_set_all = noniid(dataset_test, args.n_parties, args.shard_per_user, rand_set_all=rand_set_all)
    
    
    else:
        exit('Error: unrecognized dataset')
    traindata_cls_counts = record_net_data_stats(y_train, dict_users_train, args.logdir)
    return dataset_train, dataset_test, X_train, y_train, X_test, y_test, dict_users_train






def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


# def compute_accuracy(model, dataloader, args, n_classes, base=True, get_confusion_matrix=False, device="cpu"):
#     was_training = False
#     if model.training:
#         model.eval()
#         was_training = True

#     true_labels_list, pred_labels_list = np.array([]), np.array([])

#     correct, total = 0, 0
#     if device == 'cpu':
#         criterion = nn.CrossEntropyLoss()
#     elif "cuda" in device.type:
#         criterion = nn.CrossEntropyLoss().cuda()
#     loss_collector = []

#     with torch.no_grad():
#         for batch_idx, (x, target, _) in enumerate(dataloader):
#             if device != 'cpu':
#                 x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
#             target = target.long()
#             _, out = model(x)
#             _, pred_label = torch.max(out.data, 1)
            
#             total += x.data.size()[0]
#             if not base:
#                 target = target - n_classes
#             correct += (pred_label == target.data).sum().item()

#             if device == "cpu":
#                 pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
#                 true_labels_list = np.append(true_labels_list, target.data.numpy())
#             else:
#                 pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
#                 true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

#     if get_confusion_matrix:
#         conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

#     if was_training:
#         model.train()

#     if get_confusion_matrix:
#         return correct / float(total), conf_matrix

#     return correct / float(total)

def compute_accuracy(model, dataloader, args, n_classes, base=True, get_confusion_matrix=False, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    wrong_sample_ids = []

    correct, total = 0, 0
    sample_counter = 0  # 全局样本计数器

    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(dataloader):
            batch_size = x.size(0)
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            target = target.long()
            _, out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += batch_size
            if not base:
                target = target - n_classes
            correct += (pred_label == target).sum().item()

            pred_np = pred_label.cpu().numpy() if device != 'cpu' else pred_label.numpy()
            target_np = target.cpu().numpy() if device != 'cpu' else target.numpy()
            pred_labels_list = np.append(pred_labels_list, pred_np)
            true_labels_list = np.append(true_labels_list, target_np)

            # 错误预测样本的全局 ID 记录
            for i in range(batch_size):
                if pred_label[i] != target[i]:
                    wrong_sample_ids.append(sample_counter + i)

            sample_counter += batch_size

    if was_training:
        model.train()

    accuracy = correct / float(total)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
        return accuracy, wrong_sample_ids

    return accuracy, wrong_sample_ids



def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss



def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda()
    return model


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, domain=None):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize])




        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_base_ds = dl_obj(datadir, train=False, transform=transform_test, download=True, split='test_base')
        test_new_ds = dl_obj(datadir, train=False, transform=transform_test, download=True, split='test_new')

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True,num_workers=6, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False,num_workers=6, pin_memory=True)
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False,num_workers=6, pin_memory=True)

    elif dataset == 'tinyimagenet':
        dl_obj = TinyImageNet_load
        transform_train = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),  
            transforms.Resize(224),
            transforms.ToTensor(),                                             
            transforms.Normalize((.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_ds = dl_obj('../datasets/tiny-imagenet-200/', train=True, dataidxs=dataidxs, transform=transform_train)
        test_base_ds = dl_obj('../datasets/tiny-imagenet-200/', train=False, transform=transform_test, split='test_base')
        test_new_ds = dl_obj('../datasets/tiny-imagenet-200/', train=False, transform=transform_test, split='test_new')

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)

    
    elif dataset == 'vireo172':
        dl_obj = Vireo172_truncated
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_base_ds = dl_obj(None, transform_test, mode='test_base')
        test_new_ds = dl_obj(None, transform_test, mode='test_new')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=8, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True)          
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True)     
        
    elif dataset == 'food101':
        dl_obj = Food101_truncated
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_base_ds = dl_obj(None, transform_test, mode='test_base')
        test_new_ds = dl_obj(None, transform_test, mode='test_new')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=8, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True)          
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True)     

        
    elif dataset == 'cal101':
        dl_obj = Caltech101
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_base_ds = dl_obj(None, transform_test, mode='test_base')
        test_new_ds = dl_obj(None, transform_test, mode='test_new')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)          
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)  
        
    elif dataset == 'flowers102':
        dl_obj = Flowers102
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_base_ds = dl_obj(None, transform_test, mode='test_base')
        test_new_ds = dl_obj(None, transform_test, mode='test_new')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True) 
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)          
        
        
    elif dataset == 'air100':
        dl_obj = Air100
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_base_ds = dl_obj(None, transform_test, mode='test_base')
        test_new_ds = dl_obj(None, transform_test, mode='test_new')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True) 
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)         
                        
 
    elif dataset == 'cars196':
        dl_obj = Cars196
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_base_ds = dl_obj(None, transform_test, mode='test_base')
        test_new_ds = dl_obj(None, transform_test, mode='test_new')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)          
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)  

    elif dataset == 'ucf101':
        dl_obj = UCF101
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_base_ds = dl_obj(None, transform_test, mode='test_base')
        test_new_ds = dl_obj(None, transform_test, mode='test_new')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)          
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)  

    elif dataset == 'dtd':
        dl_obj = DTD
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_base_ds = dl_obj(None, transform_test, mode='test_base')
        test_new_ds = dl_obj(None, transform_test, mode='test_new')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True) 
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)  

    elif dataset == 'pets':
        dl_obj = Pets
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_base_ds = dl_obj(None, transform_test, mode='test_base')
        test_new_ds = dl_obj(None, transform_test, mode='test_new')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_base_dl = data.DataLoader(dataset=test_base_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True) 
        test_new_dl = data.DataLoader(dataset=test_new_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)      

    elif 'domain' in dataset:
        dl_obj = Domain_G
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        
        train_ds = dl_obj(dataset, domain=domain, transform=transform_train, dataidxs=dataidxs)
        test_base_ds = train_ds
        test_new_ds = train_ds
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_base_dl = train_dl
        test_new_dl = train_dl

    

    return train_dl, test_base_dl, test_new_dl, train_ds, test_base_ds, test_new_ds



def proto_augmentation(proto, label, args):
    
    proto_aug = []
    proto_aug_label = []
    proto = proto.to('cpu')
#     ipdb.set_trace()
    for _ in range(args.batch_size * 5):
        np.random.shuffle(label)
        temp = proto[label[0]] + torch.randn(proto[label[0]].shape) * args.radius
        proto_aug.append(temp)
        proto_aug_label.append(label[0])
#     ipdb.set_trace()
    proto_aug_list = torch.cat(proto_aug,dim=0).reshape((args.batch_size * 5, 256)) 
    proto_aug_label = torch.tensor(proto_aug_label)        
        
    return proto_aug_list, proto_aug_label

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval() 

# def gen_proto_local(net, dataloader, n_class=10, device='cuda:0'):
#     feats = []
#     labels = []
#     net.eval()
#     net.apply(fix_bn)
#     net.to(device)
#     with torch.no_grad():
#         for batch_idx, (x, target) in enumerate(dataloader):
#             x, target = x.to(device), target.to(device)
#             _, feat, _ = net(x)

#             feats.append(feat)
#             labels.extend(target)

#     feats = torch.cat(feats)
#     labels = torch.tensor(labels)
# #     ipdb.set_trace()
#     prototype = []
#     class_label = []
#     for i in range(n_class):
#         index = torch.nonzero(labels == i).reshape(-1)
#         if len(index) > 0:
#             class_label.append(int(i))
#             feature_classwise = feats[index]
#             prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))
# #     ipdb.set_trace()
#     return torch.cat(prototype, dim=0), torch.tensor(class_label)


def gen_proto_local(net, dataloader, args, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            
            feat, _ = net(x)

            feats.append(feat.cpu().numpy())
            labels.append(target.cpu().numpy())

    feats = np.concatenate(feats, 0)
    labels = np.concatenate(labels, 0)
    
    prototype = []
    class_label = []
    for i in range(n_class):
        index = np.where(labels == i)[0]
        if len(index) > 10:
            for k in range(args.num_p):
                X = feats[index]
                weights = np.random.rand(len(index))
                weights = weights / np.sum(weights)
                X_new = np.dot(weights, X)
                class_label.append(int(i))
                feature_classwise = feats[index]
                prototype.append(torch.tensor(X_new).reshape((1, -1)))
        elif len(index) >= 1:
                X = np.mean(feats[index], axis=0)
                for k in range(1):
                    disturbance = np.random.normal(0, 0.01 * np.abs(X), X.shape)
                    prototype.append(torch.tensor(X + disturbance).reshape((1, -1)))
                    class_label.append(int(i))
    return torch.cat(prototype, dim=0).cuda().half(), torch.tensor(class_label).cuda().half()
    
    # return torch.tensor(feats).cuda(), torch.tensor(labels).cuda()


def gen_proto_global(feats, labels, n_classes):
    local_proto = []
    local_labels = []
    for i in range(n_classes):
#         ipdb.set_trace()
        c_i = torch.nonzero(labels == i).reshape(-1)
        proto_i = torch.sum(feats[c_i], dim=0) / len(c_i)
        local_proto.append(proto_i.reshape(1, -1))
        local_labels.append(i)
    
    return torch.cat(local_proto, dim=0).cuda(), torch.tensor(local_labels).cuda()


def feats_ext(net, dataloader):
    imgs_feats = []
    text_feats = []
    labels = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(dataloader):
            x, target = x.cuda(), target.cuda()
            imgs_feat, text_feat, _ = net(x)
            imgs_feats.append(imgs_feat)
            if batch_idx == 0:
                text_feats.append(text_feat)
            labels.append(target)

    imgs_feats = torch.cat(imgs_feats)
    text_feats = torch.cat(text_feats)
    labels = torch.cat(labels)  

    imgs_feats = imgs_feats.cpu().numpy()
    text_feats = text_feats.cpu().numpy()
    labels = labels.cpu().numpy()
    return imgs_feats, text_feats, labels

def record_net_data(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}
    weights_cls_clients = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        weight = {unq[i]: unq_cnt[i]/np.where(y_train == i)[0].shape[0] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
        weights_cls_clients[net_i] = weight
        
#         ipdb.set_trace()

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))
    logger.info('Data statistics: %s' % str(weights_cls_clients))

    return net_cls_counts, weights_cls_clients    


def exchange_strategy(local_prototypes, local_labels, local_models, args, exchange='random'):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if exchange == 'random':
        idx_list = list(range(args.n_parties))
        random.shuffle(idx_list)
        idx_list = np.array(idx_list)
    elif exchange == 'similar':
        loss_mat = torch.zeros((args.n_parties, args.n_parties))
        for i, prototype in enumerate(local_prototypes): 
            for j in range(args.n_parties):
                model = local_models[j].cuda()
                pred = model.fc(prototype.cuda())
                loss = criterion(pred, local_labels[i]) 
                loss_mat[i][j] = loss.item()
            
        # 获取排序后的索引
        flattened_indices = torch.argsort(loss_mat.flatten())

        # 获取排序后的元素和对应的位置
        sorted_elements = loss_mat.flatten()[flattened_indices]
        row_indices, col_indices = torch.meshgrid(torch.arange(loss_mat.shape[0]), torch.arange(loss_mat.shape[1]))
        sorted_positions = torch.stack((row_indices.flatten()[flattened_indices], col_indices.flatten()[flattened_indices]), dim=1)
        
        idx_list = torch.zeros(args.n_parties) - 1
        for j in range(sorted_positions.shape[0]):
            print(sorted_positions[j][0], idx_list)
            if sorted_positions[j][0] != sorted_positions[j][1] and idx_list[sorted_positions[j][1]] == -1 and (sorted_positions[j][0] not in idx_list):
                idx_list[sorted_positions[j][1]] = sorted_positions[j][0]
            else:
                continue

            is_in_tensor = torch.any(torch.eq(idx_list, -1))

            if not is_in_tensor:
                break
        idx_list = idx_list.numpy().astype(int)
    elif exchange == 'dissimilarity':
        loss_mat = torch.zeros((args.n_parties, args.n_parties))
        for i, prototype in enumerate(local_prototypes): 
            for j in range(args.n_parties):
                model = local_models[j].cuda()
                pred = model.fc(prototype.cuda())
                loss = criterion(pred, local_labels[i]) 
                loss_mat[i][j] = loss.item()

        # 获取排序后的索引
        flattened_indices = torch.argsort(loss_mat.flatten(), descending=True)

        # 获取排序后的元素和对应的位置
        sorted_elements = loss_mat.flatten()[flattened_indices]
        row_indices, col_indices = torch.meshgrid(torch.arange(loss_mat.shape[0]), torch.arange(loss_mat.shape[1]))
        sorted_positions = torch.stack((row_indices.flatten()[flattened_indices], col_indices.flatten()[flattened_indices]), dim=1)

        idx_list = torch.zeros(args.n_parties) - 1
        for j in range(sorted_positions.shape[0]):
            print(sorted_positions[j][0], idx_list)
            if sorted_positions[j][0] != sorted_positions[j][1] and idx_list[sorted_positions[j][1]] == -1 and (sorted_positions[j][0] not in idx_list):
                idx_list[sorted_positions[j][1]] = sorted_positions[j][0]
            else:
                continue

            is_in_tensor = torch.any(torch.eq(idx_list, -1))

            if not is_in_tensor:
                break
        idx_list = idx_list.numpy().astype(int)
    return idx_list
        
        

    
def mixup_data(x, y, lam=0.1):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0.:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam    
    

    
