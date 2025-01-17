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
from function_adaptpose.model_pos_eval import evaluate

# 测分数的脚本

def main(args):
    print('==> Using settings {}'.format(args))
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_dict = data_preparation(args)

    print("==> Creating model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)

    # Check if evaluate checkpoint file exist:
    assert path.isfile(args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
    print("==> Loading checkpoint '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    try:
        model_pos.load_state_dict(ckpt['state_dict'])
    except:
        model_pos.load_state_dict(ckpt['model_pos'])

    print('==> Evaluating...')
    
    # 在H36M上测试
    # error_h36m_p1, error_h36m_p2 = evaluate(data_dict['H36M_test'], model_pos, device,pad=args.pad)
    # print('H36M: Protocol #1   (MPJPE) overall average: {:.2f} (mm)'.format(error_h36m_p1))
    # print('H36M: Protocol #2 (P-MPJPE) overall average: {:.2f} (mm)'.format(error_h36m_p2))
    
    # 在新数据集上测试
    error_3dhp_p1, error_3dhp_p2 = evaluate(data_dict['mpi3d_loader'], model_pos, device,flipaug='_flip',pad=args.pad,tag='3dhp') 
    print('测试集: Protocol #1   (MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p1))
    print('测试集: Protocol #2 (P-MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p2))


if __name__ == '__main__':
    args = get_parse_args()
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
