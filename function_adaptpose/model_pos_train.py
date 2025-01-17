from __future__ import print_function, absolute_import, division

import time

import torch
import torch.nn as nn

from progress.bar import Bar
from utils.utils import AverageMeter, set_grad


def train_posenet(model_pos, data_loader, optimizer, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    set_grad([model_pos], True)
    model_pos.train()
    end = time.time()
    
    # data_loader: data_dict['train_fake2d3d_loader']
    bar = Bar('Train posenet', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        # here avoid bn with one sample in last batch, skip if num_poses=1
        if num_poses == 1:
            break

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        if len(targets_3d.shape)>3:
            pad=(targets_3d.shape[2]-1)//2
            # targets_3d=targets_3d.squeeze()
            targets_3d=targets_3d[:,0,pad]
            
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint
        # 输入2d，过模型，跟目标3d比较
        outputs_3d = model_pos(inputs_2d)
        optimizer.zero_grad()
        # 给输出和目标打分（均方误差损失）
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return