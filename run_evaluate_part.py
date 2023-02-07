from __future__ import print_function, absolute_import, division

import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from function_adaptpose.config import get_parse_args
from function_adaptpose.data_preparation import data_preparation
from function_baseline.model_pos_preparation import model_pos_preparation
from function_adaptpose.model_pos_eval import evaluate_part


# 按照指定的方式测指标
def main(args):
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    print('==> 加载数据集...')
    data_dict = data_preparation(args)
    print('==> 数据集加载完毕...')

    print("==> 创建模型...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)
    print("==> 模型创建完毕...")

    assert path.isfile(
        args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
    print("==> 加载模型参数 '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    try:
        model_pos.load_state_dict(ckpt['state_dict'])
    except:
        model_pos.load_state_dict(ckpt['model_pos'])
    print("==> 模型参数加载完毕")

    p1_list,p2_list=evaluate_part(data_dict['mpi3d_loader'], model_pos, [[]])
    
    print(p1_list)
    # print(p2_list)

if __name__ == '__main__':
    args = get_parse_args()
    args.evaluate='checkpoint/adaptpose/videopose/gt/trained3dhp2/ckpt_best_dhp_p1.pth.tar'
    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True

    main(args)
