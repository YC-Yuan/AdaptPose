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

    print('==> Main script, data_preparation() start ...')
    data_dict = data_preparation(args)
    print('==> Main script, data_preparation() done ...')

    print("==> Main script, Creating PoseNet model start ...")
    # posenet的model以及为evaluation准备的model副本
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)
    model_pos_eval = model_pos_preparation(
        args, data_dict['dataset'], device)  # used for evaluation only
    # prepare optimizer for posenet
    # 动态调整学习率的优化器
    posenet_optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr_p)
    # 以lambda使用优化器
    posenet_lr_scheduler = get_scheduler(posenet_optimizer, policy='lambda', nepoch_fix=0,
                                         nepoch=args.epochs)

    print("==> Main script, Creating PoseAug model start ...")
    # 网络结构在此，data_dict['dataset']中是H36M数据
    poseaug_dict = get_poseaug_model(args, data_dict['dataset'])
    print("==> Main script, Creating PoseAug model done ...")

    # 均方误差损失
    criterion = nn.MSELoss(reduction='mean').to(device)

    # GAN trick: data buffer for fake data
    fake_3d_sample = Sample_from_Pool()
    fake_2d_sample = Sample_from_Pool()

    args.checkpoint = path.join(args.checkpoint, args.posenet_name, args.keypoints,
                                datetime.datetime.now().isoformat() + '_' + args.note)
    os.makedirs(args.checkpoint, exist_ok=True)
    print('==> Main script, Making checkpoint dir: {}'.format(args.checkpoint))

    # 在checkpoint文件夹打log.txt
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), args)
    logger.record_args(str(model_pos))
    logger.set_names(['Epoch', 'lr', 'H36M MPJPE',
                     'H36M P-MPJPE', '3DHP MPJPE', '3DHP P-MPJPE'])

    # Init monitor for net work training
    #########################################################
    summary = Summary(args.checkpoint)
    writer = summary.create_summary()

    ##########################################################
    # start training
    ##########################################################
    start_epoch = 0
    dhpp1_best = None  # dhp_p1
    s911p1_best = None  # h36m_p1

    continue_flag = True
    DROP_NUM = 3
    continue_count = DROP_NUM
    # 默认训练20个epoch,连续DROP_NUM个epoch没有改进之后停止
    for _ in range(start_epoch, args.epochs):
        # param init
        if summary.epoch == 0:
            poseaug_dict['optimizer_G'].zero_grad()
            poseaug_dict['optimizer_G'].step()
            poseaug_dict['optimizer_d3d'].zero_grad()
            poseaug_dict['optimizer_d3d'].step()
            poseaug_dict['optimizer_d2d'].zero_grad()
            poseaug_dict['optimizer_d2d'].step()

        #  first evaluate
        if summary.epoch == 0:
            # evaluate the pre-train model for epoch 0.
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                summary, writer)
            # h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
            #                                                   summary, writer, tag='_real')
            # h36m_p1, h36m_p2, dhp_p1, dhp_p2 = 200, 200, 200, 200
            summary.summary_epoch_update()

        if continue_flag == False:
            break  # 连续DROP_NUM个epoch没有改进就跳出

        # Train for one epoch
        # We split an epoch to five sections inorder not to face memrory problem while generating new data
        for kk in range(5):
            # 此方法传入kk,gan分五次训练
            train_gan(args, poseaug_dict, data_dict, model_pos, criterion,
                      fake_3d_sample, fake_2d_sample, summary, writer, section=kk)

            # gan训练好了再迁移训练
            if summary.epoch > args.warmup:
                train_posenet(
                    model_pos, data_dict['train_fake2d3d_loader'], posenet_optimizer, criterion, device)
                h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                    summary, writer, tag='_fake')
                # Update checkpoint
                # 每epoch更新五次
                if dhpp1_best is None or dhpp1_best > dhp_p1:
                    dhpp1_best = dhp_p1
                    logger.record_args("==> Saving checkpoint at epoch '{}', with dhp_p1 {}".format(
                        summary.epoch, dhpp1_best))
                    # 保存ckpt文件
                    continue_count = DROP_NUM
                    save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict(
                    )}, args.checkpoint, suffix='best_dhp_p1')

        # warmup默认为2，大于2后train_posenet
        if summary.epoch > args.warmup:
            train_posenet(
                model_pos, data_dict['train_gt2d3d_loader'], posenet_optimizer, criterion, device)
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                summary, writer, tag='_real')
        # Update learning rates
        poseaug_dict['scheduler_G'].step()
        poseaug_dict['scheduler_d3d'].step()
        poseaug_dict['scheduler_d2d'].step()
        posenet_lr_scheduler.step()
        lr_now = posenet_optimizer.param_groups[0]['lr']
        print('\nEpoch: %d | LR: %.8f' % (summary.epoch, lr_now))

        # 每个epoch查看效果
        if s911p1_best is None or s911p1_best > h36m_p1:
            s911p1_best = h36m_p1
            logger.record_args("==> Saving checkpoint at epoch '{}', with s911p1 {}".format(
                summary.epoch, s911p1_best))
            # 保存ckpt文件
            continue_count = DROP_NUM
            save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict(
            )}, args.checkpoint, suffix='best_h36m_p1')

        # Update log file
        logger.append([summary.epoch, lr_now, h36m_p1,
                      h36m_p2, dhp_p1, dhp_p2])

        # 决定是否还要继续训练
        continue_count -= 1
        if continue_count <= 0:
            continue_flag = False

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
