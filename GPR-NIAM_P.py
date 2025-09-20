import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tcp_clip import clip
import torch
from torch import nn
# from transformers import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from re_training import retrain_cls_relation
from tcp_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from easyfsl.samplers import TaskSampler
import numpy as np
import json
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
from utils import *
import torch.nn.functional as F
import ipdb

_tokenizer = _Tokenizer()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='federated', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.1)')
    parser.add_argument('--re_lr', type=float, default=1e-3, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=10, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="../datasets/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-3, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/TCP+LM/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--sample_fraction', type=float, default=1, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--domain_id', type=int, default=[5])
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--num_p', type=int, default=5)
    parser.add_argument('--reweight', type=float, default=0)
    args = parser.parse_args()
    return args


args = get_args()
from classes_names import *

name_classes = name_classes(args)


def load_clip_to_cpu(backbone_name):
    # backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

        self.num_heads = clip_model.transformer.resblocks[0].attn.num_heads
        self.mask_weight = args.reweight
        
    def build_text_mask(self, tokenized_prompts, seq_len_max):     
        """
        构造 attention mask：
        1. 让 learnable prompt 只能“读取”文本 token，而不会改变它们。
        2. 让每个 prompt 在 eot_token 之后的 token 被屏蔽。
        """
        attn_head = self.num_heads  # 多头注意力
        batch_size = len(tokenized_prompts)
        
        # 计算 EOT（结束标记）位置
        lengths = tokenized_prompts.argmax(dim=-1) + 1  # EOT 本身可见        
        prefix_len, n_ctx = 1, 10  # Prefix 长度 & Learnable prompt 长度        
        # 初始化 mask，全填充 -inf
        text_mask = torch.full((batch_size, seq_len_max, seq_len_max), float('-inf'), dtype=self.dtype)    
        for idx, length in enumerate(lengths):
            # 1️⃣ 允许所有 token 看到自己和之前的 token
            text_mask[idx].triu_(1)  # 设置上三角部分为 -inf (对于因果遮蔽)               
            # 2️⃣ Learnable prompt（n_ctx） 只能读取文本 token，但不能相互影响
            text_mask[idx, prefix_len:prefix_len + n_ctx, :length] = 0  # 允许 prompt 读取文本
            text_mask[idx, prefix_len + n_ctx:length, prefix_len:prefix_len + n_ctx] = float('-inf')  # 屏蔽prompt对文本的影响            
            text_mask[idx, prefix_len: prefix_len + n_ctx, prefix_len: prefix_len + n_ctx] = float('-inf')               
    
            # 3️⃣ EOT 之后的 token 全部屏蔽
            text_mask[idx, :, length:] = float('-inf')
            text_mask[idx, length-1, :prefix_len + n_ctx] = 0   
            text_mask[idx, length-1, prefix_len + n_ctx:length] = self.mask_weight   
        # 扩展到多头注意力
        return text_mask.repeat(attn_head, 1, 1).to(self.dtype)

    
    def forward(self, prompts, class_feature, weight, tokenized_prompts, flag=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        text_mask = self.build_text_mask(tokenized_prompts, x.shape[0])
        if flag:
            x = self.transformer(x, text_mask)
        else:
            counter = 0
            for block in self.transformer.resblocks:
                outputs = block([x, class_feature, weight, counter], text_mask)
                x = outputs[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


CUSTOM_TEMPLATES_ori = {
    "pets": "a photo of a {}, a type of pet.",
    "flowers102": "a photo of a {}, a type of flower.",
    "air100": "a photo of an aircraft {}.",
    "dtd": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "cars196": "a photo of a {}.",
    "Food101": "a photo of a {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "cal101": "a photo of a {}.",
    "ucf101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "domain_vlcs": "a photo of a {}.",
    "domain_office": "a photo of a {}.",
    "domain_net": "a photo of a {}.",
    # Cifar10 and Cifar100
    "cifar10": "a photo of a {}.",
    "cifar100": "a photo of a {}.",
}

CUSTOM_TEMPLATES = {
    "pets": "X X X X X X X X X X a photo of a {}, a type of pet.",
    "flowers102": "X X X X X X X X X X a photo of a {}, a type of flower.",
    "air100": "X X X X X X X X X X a photo of a aircraft {}.",
    "dtd": "X X X X X X X X X X a photo of a {}.",
    "EuroSAT": "X X X X X X X X X X {}.",
    "cars196": "X X X X X X X X X X a photo of a {}.",
    "Food101": "X X X X X X X X X X {}, a type of food.",
    "SUN397": "X X X X X X X X X X a photo of a {}.",
    "cal101": "X X X X X X X X X X a photo of a {}.",
    "ucf101": "X X X X X X X X X X a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "domain_vlcs": "X X X X X X X X X X a photo of a {}.",
    "domain_office": "X X X X X X X X X X {}.",    
    "domain_net": "X X X X X X X X X X a photo of a {}.",
    # Cifar10 and Cifar100
    "cifar10": "X X X X X X X X X X a photo of a {}.",
    "cifar100": "X X X X X X X X X X a photo of a {}.",
}


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 10  # cfg.TRAINER.COOP.N_CTX
        ctx_init = False  # cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224  # cfg.INPUT.SIZE[0]
        csc = False  # cfg.TRAINER.COOP.CSC:class-specific context
        prec = "fp16"  # "fp32" "amp" cfg.TRAINER.COCOOP.PREC
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            print("use given words to initialize context vectors")
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

            ctx_vectors_src = embedding[0, 1: 1 + n_ctx, :]

        else:
            # random initialization
            if csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        clip_model_ = load_clip_to_cpu("ViT-B/16")
        clip_model_.cuda()
        #################
        ##需要再修改提示词##
        ################
        temp = CUSTOM_TEMPLATES_ori[args.dataset]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        vis_dim = clip_model.visual.output_dim
        self.meta_net = nn.Sequential(
            OrderedDict([("linear1", nn.Linear(vis_dim, vis_dim // 4, bias=True)),
                         ("relu", QuickGELU()),
                         ("linear2", nn.Linear(vis_dim // 4, 4 * ctx_dim, bias=True))
                         ]))
        if prec == "fp16":
            self.meta_net.half()
        classnames = [name.replace("_", " ") for name in classnames]
        temp = CUSTOM_TEMPLATES[args.dataset]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(prompts)
        # ipdb.set_trace()
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = "end"  # 'middle' or 'end' or 'front' cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.prev_ctx = None

    def forward(self):
        class_feature = self.meta_net(self.text_features)
        class_feature = class_feature.reshape(class_feature.shape[0], -1, 512)
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompt = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompt, class_feature


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, args)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.domain_sim = -1
        self.domain_sim_src = -1
        self.weight = 0.5  # cfg.TRAINER.COOP.W

    def forward_re(self, image_features):
        text_features_old = self.ori_embedding
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, class_prompt = self.prompt_learner()
        text_features = self.text_encoder(prompts, class_prompt, self.weight, tokenized_prompts.detach())
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale.detach() * image_features.detach() @ text_features_norm.t()

        if self.prompt_learner.training:
            return text_features_norm, text_features_old, logits
        else:
            return image_features, logits

    
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        text_features_old = self.ori_embedding
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, class_prompt = self.prompt_learner()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_encoder(prompts, class_prompt, self.weight, tokenized_prompts.detach())
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale.detach() * image_features.detach() @ text_features_norm.t()

        if self.prompt_learner.training:
            return text_features_norm, text_features_old, logits
        else:
            return image_features, logits


def init_nets(net_configs, n_parties, args, n_classes, net_classname_map, domains_list=None, device='cuda:0'):
    #     clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model = load_clip_to_cpu("ViT-B/16")
    nets = {net_i: None for net_i in range(n_parties)}

    train_datasets = []
    for net_i in range(n_parties):
        name_classes_i = net_classname_map[net_i]
        if "domain" in args.dataset:
            n_i = args.n_parties // len(domains_list)
            print("nnnnnnnnn", net_i, n_i)
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])
            d_i = net_i // n_i
            print('ddddddddd', d_i)
            train_ds = Domain_G(args.dataset, domain_lists[d_i], transform_train)
            fold_idx = net_i % n_i
            print('fffffffff', fold_idx)
            train_data, train_labels, m_labels = train_ds.folds[fold_idx]
            train_datasets.append((train_data, train_labels, m_labels))
            name_classes_i = [name_classes[i] for i in set(train_labels)]
        net = CustomCLIP(name_classes_i, clip_model)
        for name, param in net.named_parameters():
            if "prompt_learner.ctx" not in name and "prompt_learner.meta" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        nets[net_i] = net.to(device)

    log_model_info(net)

    global_net = CustomCLIP(name_classes[:n_classes], clip_model)

    if "domain" in args.dataset:
        global_net_new = CustomCLIP(name_classes, clip_model)
    else:
        global_net_new = CustomCLIP(name_classes[n_classes:], clip_model)

    for (name, param), (name_new, param_new) in zip(global_net.named_parameters(), global_net_new.named_parameters()):
        if "prompt_learner.ctx" not in name and "prompt_learner.meta" not in name:
            param.requires_grad_(False)
            param_new.requires_grad_(False)
        else:
            param.requires_grad_(True)
            param_new.requires_grad_(True)
            print(name)
    # ipdb.set_trace()

    if "domain" in args.dataset:
        return nets, global_net, global_net_new, train_datasets
    else:
        return nets, global_net, global_net_new


def train_net_fedavg(net_id, net, train_dataloader, epochs, lr, args_optimizer, args, device="cuda:0"):
    net.to(device)

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, _, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            if args.dataset == 'pmnist':
                target = target.reshape(-1)
            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()
            text_features_norm, text_features_old, out = net(x)
            score = cos(text_features_norm, text_features_old)
            score = 1.0 - torch.mean(score)
            loss = criterion(out, target) + 8.0 * score
            # ipdb.set_trace()
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0


def local_train_net(nets, args, net_dataidx_map, train_dl=None, global_model=None, prev_model_pool=None,
                    prev_protos_pool=None, prev_protos_label_pool=None, server_c=None, clients_c=None, round=None,
                    domain_lists=None, device="cuda:0"):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)

    Anchors, Anchor_labels = [], []
    for net_id, net in nets.items():

        # 区分域泛化和非独立同分布的数据
        if 'domain' not in args.dataset:
            dataidxs = net_dataidx_map[net_id]
            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            if len(dataidxs) > args.batch_size:
                train_dl_local, test_dl_local, _, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            else:
                train_dl_local, test_dl_local, _, _, _, _ = get_dataloader(args.dataset, args.datadir, len(dataidxs)-1, 32, dataidxs)
        else:
            n_i = args.n_parties // len(domain_lists)
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])
            train_data, train_labels, m_labels = train_dl[net_id]
            train_dataset = CustomDataset(train_data, m_labels, transform=transform_train)
            # 创建 DataLoader
            train_dl_local = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        n_epoch = args.epochs
        trainacc, testacc = train_net_fedavg(net_id, net, train_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)   
        local_protos, local_labels = gen_proto_local(net, train_dl_local, args, n_class=n_classes)
        Anchors.append(local_protos)
        Anchor_labels.extend(local_labels)
        
    Anchors = torch.cat(Anchors, dim=0)
    Anchor_labels = torch.tensor(Anchor_labels)
    
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets, Anchors, Anchor_labels    


if __name__ == '__main__':
    if 'domain' in args.dataset:
        args.logdir = args.logdir + args.dataset + '/' + str(args.domain_id)
    else:
        args.logdir = args.logdir + args.dataset + '/' + str(args.beta)+ '/' + str(args.epochs) + '/clients_' + str(args.n_parties)
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = str(args.num_p) + '_' + str(args.reweight) + '_' + args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = str(args.num_p) + '_' + str(args.reweight) + '_' + args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")

    if 'domain' not in args.dataset:
        X_train, y_train, X_base_test, y_base_test, X_new_test, y_new_test, net_dataidx_map, net_classname_map, traindata_cls_counts = partition_data(
            args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, name_classes, beta=args.beta)
        n_classes = len(np.unique(y_train))

        train_dl_global, test_base_dl, test_new_dl, train_ds_global, test_base_ds_global, test_new_ds_global = get_dataloader(
            args.dataset, args.datadir,
            args.batch_size, 32)

        print("len train_dl_global:", len(train_ds_global))
        copy_domains_list = None

    else:
        if 'domain_office' == args.dataset:
            n_classes = 65
            domain_lists = ['Art', 'Clipart', 'Product', 'RealWorld']
        elif 'domain_vlcs' == args.dataset:
            n_classes = 5
            domain_lists = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
        elif 'domain_pacs' == args.dataset:
            n_classes = 7
            domain_lists = ['art_painting', 'cartoon', 'photo', 'sketch']
        elif 'domain_net' == args.dataset:
            n_classes = 20
            domain_lists = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

        copy_domains_list = copy.deepcopy(domain_lists)
        copy_domains_list = [v for i, v in enumerate(copy_domains_list) if i not in args.domain_id]
        net_classname_map = [name_classes for i in range(args.n_parties)]
        net_dataidx_map = None

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    global_party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    train_dl = None
    #     data_size = len(test_ds_global)

    logger.info("Initializing nets")
    if "domain" in args.dataset:
        nets, global_model, global_model_new, train_dl = init_nets(args.net_config, args.n_parties, args, n_classes, net_classname_map, domains_list=copy_domains_list, device=device)
    else:
        nets, global_model, global_model_new = init_nets(args.net_config, args.n_parties, args, n_classes, net_classname_map, domains_list=copy_domains_list, device=device)
    for name, param in global_model.state_dict().items():
        print(name, param.shape)

    # ipdb.set_trace()

    n_comm_rounds = args.comm_round

    if args.mode == "center":
        n_epoch = 30
        trainacc, testacc = train_net_fedavg(0, global_model, train_dl_global, n_epoch, args.lr, args.optimizer, args,
                                             device=device)

    else:
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            nets_this_round = {k: nets[k] for k in party_list_this_round}

            global_w = global_model.state_dict()

            for net in nets_this_round.values():
                net_dict = net.state_dict()
                shared_dict = {k: v for k, v in global_w.items() if ("prompt_learner.ctx" in k or "prompt_learner.meta" in k)}
                net_dict.update(shared_dict)
                net.load_state_dict(net_dict)

            nets_this_round, Anchors, Anchor_labels = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,
                                              global_model=global_model, round=round, domain_lists=copy_domains_list,
                                              device=device)

            global_model.to('cpu')

            # update global model
            global_nets_this_round = {k: nets[k] for k in global_party_list}
            if 'domain' not in args.dataset:
                total_data_points = sum([len(net_dataidx_map[r]) for r in global_party_list])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in global_party_list]
            else:
                fed_avg_freqs = [1 / args.n_parties] * args.n_parties

            global_model_new.to('cpu')
            global_w_new = global_model_new.state_dict()
            for net_id, net in enumerate(global_nets_this_round.values()):
                net.to('cpu')
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        if "prompt_learner.ctx" in key or "prompt_learner.meta" in key:
                            global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                            global_w_new[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        if "prompt_learner.ctx" in key or "prompt_learner.meta" in key:
                            global_w[key] += net_para[key] * fed_avg_freqs[net_id]
                            global_w_new[key] += net_para[key] * fed_avg_freqs[net_id]

            global_model.load_state_dict(global_w)
            global_model_new.load_state_dict(global_w_new)

            global_model.cuda()
            global_model_new.cuda()

            if 'domain' not in args.dataset:
                test_base_acc, conf_matrix = compute_accuracy(global_model, test_base_dl, args, n_classes, get_confusion_matrix=True, device=device)
                test_new_acc, conf_matrix = compute_accuracy(global_model_new, test_new_dl, args, n_classes, base=False, get_confusion_matrix=True, device=device)
                Harmonic_mean = (test_base_acc * test_new_acc) * 2 / (test_base_acc + test_new_acc) 
    
        #         logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Base Test accuracy: %f' % test_base_acc)
                logger.info('>> Global Model New Test accuracy: %f' % test_new_acc)
                logger.info('>> Global Model Harmonic mean: %f' % Harmonic_mean)

                global_model = retrain_cls_relation(global_model, Anchors, Anchor_labels, args, round, 'cuda:0')
                
                global_w = global_model.state_dict()
                for key in net_para:
                    if key == "prompt_learner.ctx":
                        global_w_new[key] = global_w[key]
                global_model_new.load_state_dict(global_w_new)
                
                global_model.cuda()
                global_model_new.cuda()
                test_base_acc, conf_matrix = compute_accuracy(global_model, test_base_dl, args, n_classes, get_confusion_matrix=True, device=device)
                test_new_acc, conf_matrix = compute_accuracy(global_model_new, test_new_dl, args, n_classes, base=False, get_confusion_matrix=True, device=device)
                Harmonic_mean = (test_base_acc * test_new_acc) * 2 / (test_base_acc + test_new_acc) 
    
        #         logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global After Model Base Test accuracy: %f' % test_base_acc)
                logger.info('>> Global After Model New Test accuracy: %f' % test_new_acc)
                logger.info('>> Global After Model Harmonic mean: %f' % Harmonic_mean)     

            else:
                normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
                 ])            

                for idd in args.domain_id:
                    test_ds = Domain_G(args.dataset, domain_lists[idd], transform_test)
                    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
                    test_base_acc, conf_matrix = compute_accuracy(global_model, test_dl, args, n_classes, get_confusion_matrix=True, device=device)
    
                    logger.info('>> Global Model Leave-One-Domain-Out Test accuracy: %f, Domain ID: %s' % (test_base_acc, str(args.domain_id)))
                    np.set_printoptions(threshold=np.inf)  # 让 NumPy 完全显示数组
                    logger.info('>> Global Model Leave-One-Domain-Out Test conf_matrix: %s, Domain ID: %s' % (str(conf_matrix), str(args.domain_id)))
                    # torch.save({'prompt_ctx': global_model.prompt_learner.ctx}, "CoOp_prompt_ctx.pth")

                    # retrain
                    global_model = retrain_cls_relation(global_model, Anchors, Anchor_labels, args, round, 'cuda:0')
                    test_base_acc, conf_matrix = compute_accuracy(global_model, test_dl, args, n_classes, get_confusion_matrix=True, device=device)
    
                    logger.info('>> Global Model Leave-One-Domain-Out Test accuracy: %f, Domain ID: %s' % (test_base_acc, str(args.domain_id)))
                    np.set_printoptions(threshold=np.inf)  # 让 NumPy 完全显示数组
                    logger.info('>> Global Model Leave-One-Domain-Out Test conf_matrix: %s, Domain ID: %s' % (str(conf_matrix), str(args.domain_id)))
            
