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
import torch.optim as optim
# from model import *
from datasets import *
import ipdb
import copy
import torch

  
def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    # 每四次epoch调整一下lr，将lr减半
    lr = init_lr * (decay_rate ** (epoch // lr_decay))  # *是乘法，**是乘方，/是浮点除法，//是整数除法，%是取余数

    if epoch % lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 返回改变了学习率的optimizer
    return optimizer




def retrain_cls_relation(model, prototypes, proto_labels, args, round, device):

    init_lr = args.re_lr
    model.to(device)
    model.train()
    lr_decay = 15
    decay_rate = 0.1

    for name, param in model.named_parameters():
        if name not in ["prompt_learner.ctx", "prompt_learner.meta"]:
            param.requires_grad_(False)    
        else:
            param.requires_grad_(True)
        
    prototypes = prototypes.to(device)
    proto_labels = proto_labels.to(device)
    # ipdb.set_trace()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    idx_list = np.array(np.arange(len(proto_labels)))
    batch_size = 64
    # ipdb.set_trace()   
    for epoch in range(10):
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate)
        random.shuffle(idx_list)
        
        epoch_celoss_collector=[]   

        for i in range(len(proto_labels)//batch_size+1):
            if i < len(proto_labels)//batch_size:
                x = prototypes[idx_list[i*batch_size:(i+1)*batch_size]]
                target = proto_labels[idx_list[i*batch_size:(i+1)*batch_size]]                                 
                
            optimizer.zero_grad()
            target = target.long()         
           
            _, _, out = model.forward_re(x)
            loss = criterion(out, target)

            # with torch.no_grad():
            #     pass
            
            loss.backward()
            optimizer.step()
            epoch_celoss_collector.append(loss.item())
            
#         ipdb.set_trace()
        print(epoch, sum(epoch_celoss_collector)/len(epoch_celoss_collector))

    return model


