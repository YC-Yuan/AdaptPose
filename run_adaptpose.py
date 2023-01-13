from __future__ import print_function, absolute_import, division

import datetime
import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from function_baseline.model_pos_preparation import model_pos_preparation
from function_adaptpose.config import get_parse_args
from function_adaptpose.data_preparation import data_preparation
from function_adaptpose.dataloader_update import dataloader_update
from function_adaptpose.model_gan_preparation import get_poseaug_model
from function_adaptpose.model_gan_train import train_gan
from function_adaptpose.model_pos_eval import evaluate_posenet
from function_adaptpose.model_pos_train import train_posenet
from utils.gan_utils import Sample_from_Pool
from utils.log import Logger
from utils.utils import save_ckpt, Summary, get_scheduler


def main(args):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_dict = data_preparation(args)

    print("==> Creating PoseNet model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)
    model_pos_eval = model_pos_preparation(args, data_dict['dataset'], device)  # used for evaluation only
    # prepare optimizer for posenet
    posenet_optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr_p)
    posenet_lr_scheduler = get_scheduler(posenet_optimizer, policy='lambda', nepoch_fix=0,
                                         nepoch=args.epochs)

    print("==> Creating PoseAug model...")
    poseaug_dict = get_poseaug_model(args, data_dict['dataset'])

    # loss function
    criterion = nn.MSELoss(reduction='mean').to(device)

    # GAN trick: data buffer for fake data
    fake_3d_sample = Sample_from_Pool()
    fake_2d_sample = Sample_from_Pool()

    args.checkpoint = path.join(args.checkpoint, args.posenet_name, args.keypoints,
                              datetime.datetime.now().isoformat() + '_' + args.note)
    os.makedirs(args.checkpoint, exist_ok=True)
    print('==> Making checkpoint dir: {}'.format(args.checkpoint))

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), args)
    logger.record_args(str(model_pos))
    logger.set_names(['epoch', 'lr', 'error_h36m_p1', 'error_h36m_p2', 'error_3dhp_p1', 'error_3dhp_p2'])

    # Init monitor for net work training
    #########################################################
    summary = Summary(args.checkpoint)
    writer = summary.create_summary()

    ##########################################################
    # start training
    ##########################################################
    start_epoch = 0
    dhpp1_best = None
    s911p1_best = None

    # 默认训练50个epoch
    for _ in range(start_epoch, args.epochs):
        # epoch0的参数准备
        if summary.epoch == 0:
            poseaug_dict['optimizer_G'].zero_grad()
            poseaug_dict['optimizer_G'].step()
            poseaug_dict['optimizer_d3d'].zero_grad()
            poseaug_dict['optimizer_d3d'].step()
            poseaug_dict['optimizer_d2d'].zero_grad()
            poseaug_dict['optimizer_d2d'].step()
        if summary.epoch == 0:
            # evaluate the pre-train model for epoch 0.
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                  summary, writer, tag='_fake')
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                  summary, writer, tag='_real')
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 =200,200,200,200
            summary.summary_epoch_update()
      
        # Train for one epoch
        # We split an epoch to five sections inorder not to face memrory problem while generating new data
        for kk in range(5):
            # 此方法传入kk,gan分五次训练
            train_gan(args, poseaug_dict, data_dict, model_pos, criterion, fake_3d_sample, fake_2d_sample, summary, writer, section=kk)

            # warmup默认为2，大于2后train_posenet
            if summary.epoch > args.warmup:
                train_posenet(model_pos, data_dict['train_fake2d3d_loader'], posenet_optimizer, criterion, device)
                h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                    summary, writer, tag='_fake')
                # Update checkpoint
                # 每epoch更新五次
                if dhpp1_best is None or dhpp1_best > dhp_p1:
                    dhpp1_best = dhp_p1
                    logger.record_args("==> Saving checkpoint at epoch '{}', with dhp_p1 {}".format(summary.epoch, dhpp1_best))
                    save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict()}, args.checkpoint, suffix='best_dhp_p1')

        # warmup默认为2，大于2后train_posenet
        if summary.epoch > args.warmup:
            train_posenet(model_pos, data_dict['train_gt2d3d_loader'], posenet_optimizer, criterion, device)
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                               summary, writer, tag='_real')
        # Update learning rates
        ########################
        poseaug_dict['scheduler_G'].step()
        poseaug_dict['scheduler_d3d'].step()
        poseaug_dict['scheduler_d2d'].step()
        posenet_lr_scheduler.step()
        lr_now = posenet_optimizer.param_groups[0]['lr']
        print('\nEpoch: %d | LR: %.8f' % (summary.epoch, lr_now))

        # Update log file
        logger.append([summary.epoch, lr_now, h36m_p1, h36m_p2, dhp_p1, dhp_p2])

        # 每个epoch查看效果
        if s911p1_best is None or s911p1_best > h36m_p1:
            s911p1_best = h36m_p1
            logger.record_args("==> Saving checkpoint at epoch '{}', with s911p1 {}".format(summary.epoch, s911p1_best))
            save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict()}, args.checkpoint, suffix='best_h36m_p1')

        summary.summary_epoch_update()

    writer.close()
    logger.close()


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
