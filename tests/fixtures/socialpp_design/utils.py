# -*- coding: utf-8 -*-
# @Time   : 2020/7/17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2021/3/8
# @Author : Jiawei Guan
# @Email  : guanjw@ruc.edu.cn

"""
recbole.utils.utils
################################
"""

import datetime
import importlib
import logging
import os
import random
from pytz import timezone

import numpy as np
import torch
import scipy.sparse as sp


def normalize_sp(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten() # .flatten convert the array into 1-dim
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_dense(mx):
    """Row-normalize dense matrix"""
    # adj 行正则 x/1e-12=0 因为1e-12对应的那些x都是0
    mx = mx / np.clip(np.sum(mx, axis=-1, keepdims=True), a_min=1e-12, a_max=None) # VERY_SMALL_NUMBER =1e-12
    return mx

def random_neq(l, r, pos_set):
    # 随机选择一个不在交互序列中的作为负样本
    t = np.random.randint(l, r)
    while t in pos_set:
        t = np.random.randint(l, r)
    return t

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total parameter': total_num, 'Trainable parameter': trainable_num}

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now(timezone('Asia/Shanghai'))
    cur = cur.strftime('%b-%d-%Hh%Mm%Ss')

    return cur


def mkdir_ifnotexist(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def early_stopping(value, best, cur_step, max_step, bigger=False):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:  # 如果想让指标越大越好 则>  并且best初始化为很小的值     如果想让指标越小越好的话 则<    并且best初始化为很大
            # 的值
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r""" return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result['Recall@10']


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ' : ' + str(value) + '    '
    return result_str


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False




def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)


def initLogging(output_dir):
    """Init for logging
    """
    # t = get_local_time()
    # output_dir =  f"./log/{args.dataset}" # 模型超参数以及结果保存
    mkdir_ifnotexist(output_dir)
    mkdir_ifnotexist('./saved')
    # log_dir = os.path.join(output_dir, f'log_{t}.txt')

    
    path = output_dir
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logFilename = os.path.join(path, current_time+'.txt')

    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    # ref: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def print_config(config):
    print('\n[INFO] config')
    for k in config.keys():
        print(f'{k}: {config[k]}')
    print('---'*5)


    
# def full_inference(valid_data, train_data, config, valid_loader):
#     rec_result = {}
#     users_pos_items, train_pos_len_list = get_pos_items_per_user(valid_data, train_data, config)

#     with torch.no_grad():
#         positem_pt, vuser_pt = 0, 0 # (1) positem_pt: item point in users_pos_items; (2) vuser_pt: user point in train_pos_len_list
#         for bidx, (user, pos, negs) in enumerate(tqdm(valid_loader, file=sys.stdout)):
#             this_bs = user.shape[0]
#             num_pos_items = sum(train_pos_len_list[vuser_pt: vuser_pt+this_bs]) # get the num of pos items this batch user interacted
#             batch_mask = users_pos_items[:, positem_pt: positem_pt+num_pos_items]
#             # print(batch_mask)
#             batch_mask[0] -= vuser_pt # 在batch中的相对位置
#             positem_pt += num_pos_items
#             vuser_pt += this_bs

#             user = user.to(device)  # [B]
#             # 不同的inference方式
#             # batch_scores = net.full_query(user, method=1) # social only
#             batch_scores = net.full_query(user, method=0) # inference from uu+ui graph

#             # print(batch_scores.shape)
#             # print(batch_mask, batch_mask[0].max(), batch_mask[1].max())
#             batch_scores[batch_mask[0], batch_mask[1]] = -1e10 # 把batch中的user交互过的pos item给mask掉
#             _, topk_idx = torch.topk(batch_scores, 30, dim=-1)
#             # print(topk_idx, topk_idx.shape)
#             # print(pos)
#             for idx, u in enumerate(user):
#                 u_number = int(u)+1
#                 topk_li = [int(ki) for ki in topk_idx[idx]]
#                 rec_result[u_number] = topk_li
    
#         for uu in range(num_of_target_users+1, num_of_all_users+1):
#             # print(uu)
#             uu = torch.tensor(uu-1)
#             uu = uu.to(device)
#             batch_scores = net.full_query(uu, method=0)
#             _, topk_idx = torch.topk(batch_scores, 30, dim=-1)
#             topk_li = [int(ki) for ki in topk_idx]
#             rec_result[int(uu)+1] = topk_li

#     return rec_result