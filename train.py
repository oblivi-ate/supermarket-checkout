#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超市商品识别模型训练脚本
支持YOLOv7模型的自定义训练
"""

import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from src.models.experimental import attempt_load
from src.models.yolo import Model
from src.utils.autoanchor import check_anchors
from src.utils.datasets import create_dataloader
from src.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from src.utils.google_utils import attempt_download
from src.utils.loss import ComputeLoss
from src.utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from src.utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from src.utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)

def train(hyp, opt, device, tb_writer=None):
    """
    训练函数
    """
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # 目录设置
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # 创建权重目录
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # 保存运行设置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # 配置
    plots = not opt.evolve  # 创建图表
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # 数据字典
    is_coco = opt.data.endswith('coco.yaml')

    # 日志记录
    loggers = {'wandb': None}  # 日志记录器字典
    if rank in [-1, 0]:
        opt.hyp = hyp  # 添加超参数
        run_id = torch.randn(1).item() if opt.resume else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger可能更新这些值

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # 类别数量
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # 类别名称
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # 检查

    # 模型
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # 下载权重文件
        ckpt = torch.load(weights, map_location=device)  # 加载检查点
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # 排除键
        state_dict = ckpt['model'].float().state_dict()  # 转换为FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # 相交
        model.load_state_dict(state_dict, strict=False)  # 加载
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # 报告
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建

    # 冻结层
    freeze = []  # 要冻结的参数名称（完全匹配）
    for k, v in model.named_parameters():
        v.requires_grad = True  # 训练所有层
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # 优化器
    nbs = 64  # 标称批量大小
    accumulate = max(round(nbs / total_batch_size), 1)  # 累积损失之前的批次数
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # 缩放weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # 优化器参数组
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # 偏置
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # 无衰减
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # 应用衰减

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # 调整beta1到momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # 添加pg1（权重）
    optimizer.add_param_group({'params': pg2})  # 添加pg2（偏置）
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # 调度器 https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # 线性
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # 余弦 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # 恢复
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # 优化器
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # 结果
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # 写入results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # 微调额外的epochs

        del ckpt, state_dict

    # 图像大小
    gs = max(int(model.stride.max()), 32)  # 网格大小（最大步长）
    nl = model.model[-1].nl  # 检测层数量（用于缩放hyp['obj']）
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 验证imgsz是gs的倍数

    # DP模式
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # 训练加载器
    dataloader, dataset = create_dataloader(data_dict['train'], imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=rank, world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 最大标签类别
    nb = len(dataloader)  # 批次数量
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # 进程0
    if rank in [-1, 0]:
        testloader = create_dataloader(data_dict['val'], imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # 类别
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # 频率
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # 锚点
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # 预减少锚点精度

    # DDP模式
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # 模型参数
    hyp['box'] *= 3. / nl  # 缩放到层数
    hyp['cls'] *= nc / 80. * 3. / nl  # 缩放到类别和层数
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # 缩放到图像大小和层数
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # 将nc附加到模型
    model.hyp = hyp  # 将超参数附加到模型
    model.gr = 1.0  # iou损失比率（obj_loss = 1.0或iou）
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 将类别权重附加到模型
    model.names = names

    # 开始训练
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # 预热迭代次数，max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # 限制预热到< 1/2的训练
    maps = np.zeros(nc)  # 每个类别的mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # 不移动
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # 初始化损失类
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    
    # 训练循环开始
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # 更新图像权重（可选）
        if opt.image_weights:
            # 根据前一个epoch的结果生成权重
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # 类别权重
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # 图像权重
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # 随机加权索引
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # 更新马赛克边框
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # 平均损失
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # 进度条
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # 集成批次数量（自训练开始）
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8到float32，0-255到0.0-1.0

            # 预热
            if ni <= nw:
                xi = [0, nw]  # x插值
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou损失比率（obj_loss = 1.0或iou）
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # 偏置lr从0.1降到lr0，所有其他参数从0上升到lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # 多尺度
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # 大小
                sf = sz / max(imgs.shape[2:])  # 缩放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # 新形状（拉伸到gs的倍数）
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # 前向传播
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # 前向传播
                loss, loss_items = compute_loss(pred, targets.to(device))  # 损失缩放由GradScaler处理
                if rank != -1:
                    loss *= opt.world_size  # 梯度平均DDP训练损失
                if opt.quad:
                    loss *= 4.

            # 反向传播
            scaler.scale(loss).backward()

            # 优化
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # 打印
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # 绘制
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # 文件名
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # 添加模型图
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # 批次结束 ------------------------------------------------------------------------------------------------------------
        # epoch结束 ------------------------------------------------------------------------------------------------------------

        # 调度器
        lr = [x['lr'] for x in optimizer.param_groups]  # 用于tensorboard
        scheduler.step()

        # DDP进程0或单GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # 计算mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco)

            # 写入
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # 追加指标，val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # 日志
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # 训练损失
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val损失
                    'x/lr0', 'x/lr1', 'x/lr2']  # 参数
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # 更新最佳mAP
            fi = fitness(np.array(results).reshape(1, -1))  # 加权组合[P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # 保存模型
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # 如果save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # 保存last, best和delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs-5):
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # epoch结束 ------------------------------------------------------------------------------------------------------------
    # 训练结束

    if rank in [-1, 0]:
        # 绘制
        if plots:
            plot_results(save_dir=save_dir)  # 保存results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})

        # 测试最佳模型
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # 如果是COCO数据集
            for m in [last, best] if best.exists() else [last]:  # 速度，mAP测试
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco)

        # 剥离优化器
        final = best if best.exists() else last  # 最终模型
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # 剥离优化器
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # 上传
        if wandb_logger.wandb and not opt.evolve:  # 记录最终模型
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                             name='run_' + wandb_logger.wandb_run.id + '_model',
                                             aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov7.pt', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default='', help='模型.yaml路径')
    parser.add_argument('--data', type=str, default='data/supermarket.yaml', help='数据.yaml路径')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='超参数路径')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='总批量大小适用于所有GPU')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test]图像大小')
    parser.add_argument('--rect', action='store_true', help='矩形训练')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复最近的训练')
    parser.add_argument('--nosave', action='store_true', help='仅保存最终检查点')
    parser.add_argument('--notest', action='store_true', help='仅在最终epoch测试')
    parser.add_argument('--noautoanchor', action='store_true', help='禁用autoanchor检查')
    parser.add_argument('--evolve', action='store_true', help='进化超参数')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='缓存图像以加快训练速度')
    parser.add_argument('--image-weights', action='store_true', help='使用加权图像选择进行训练')
    parser.add_argument('--device', default='', help='cuda设备，即0或0,1,2,3或cpu')
    parser.add_argument('--multi-scale', action='store_true', help='变化img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='训练多类数据作为单类')
    parser.add_argument('--adam', action='store_true', help='使用torch.optim.Adam()优化器')
    parser.add_argument('--sync-bn', action='store_true', help='使用SyncBatchNorm，仅在DDP模式下可用')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP参数，不要修改')
    parser.add_argument('--workers', type=int, default=8, help='最大数据加载器工作进程数')
    parser.add_argument('--project', default='runs/train', help='保存到project/name')
    parser.add_argument('--entity', default=None, help='W&B实体')
    parser.add_argument('--name', default='exp', help='保存到project/name')
    parser.add_argument('--exist-ok', action='store_true', help='现有project/name确定，不增量')
    parser.add_argument('--quad', action='store_true', help='四数据加载器')
    parser.add_argument('--linear-lr', action='store_true', help='线性LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='上传数据集作为W&B工件表')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='设置边界框图像记录间隔为W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='记录模型后每x epochs（禁用为-1）')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='要使用的数据集工件的版本')
    opt = parser.parse_args()

    # 设置DDP变量
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1))
    opt.global_rank = int(os.environ.get('RANK', -1))
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()
        check_requirements()

    # 恢复
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # 恢复中断的运行
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # 指定或最近的路径
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml', encoding='utf-8') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # 替换
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # 重新分配
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # 检查文件
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # 扩展到2个大小（train, test）
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # 增量运行

    # DDP模式
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # 分布式后端
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # 超参数
    with open(opt.hyp, encoding='utf-8') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # 加载超参数

    # 训练
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # 进化超参数（可选）
    else:
        # 超参数进化元数据（突变比例0-1，下限，上限）
        meta = {'lr0': (1, 1e-5, 1e-1),  # 初始学习率（SGD=1E-2，Adam=1E-3）
                'lrf': (1, 0.01, 1.0),  # 最终OneCycleLR学习率（lr0 * lrf）
                'momentum': (0.3, 0.6, 0.98),  # SGD动量/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # 优化器权重衰减
                'warmup_epochs': (1, 0.0, 5.0),  # 预热epochs（分数ok）
                'warmup_momentum': (1, 0.0, 0.95),  # 预热初始动量
                'warmup_bias_lr': (1, 0.0, 0.2),  # 预热初始偏置lr
                'box': (1, 0.02, 0.2),  # 框损失增益
                'cls': (1, 0.2, 4.0),  # cls损失增益
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss正权重
                'obj': (1, 0.2, 4.0),  # obj损失增益（按像素缩放）
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss正权重
                'iou_t': (0, 0.1, 0.7),  # IoU训练阈值
                'anchor_t': (1, 2.0, 8.0),  # 锚点倍数阈值
                'anchors': (2, 2.0, 10.0),  # 每个输出网格的锚点（0忽略）
                'fl_gamma': (0, 0.0, 2.0),  # 焦点损失gamma（efficientDet默认gamma=1.5）
                'hsv_h': (1, 0.0, 0.1),  # 图像HSV-Hue增强（分数）
                'hsv_s': (1, 0.0, 0.9),  # 图像HSV-Saturation增强（分数）
                'hsv_v': (1, 0.0, 0.9),  # 图像HSV-Value增强（分数）
                'degrees': (1, 0.0, 45.0),  # 图像旋转（+/-度）
                'translate': (1, 0.0, 0.9),  # 图像平移（+/-分数）
                'scale': (1, 0.0, 0.9),  # 图像缩放（+/-增益）
                'shear': (1, 0.0, 10.0),  # 图像剪切（+/-度）
                'perspective': (0, 0.0, 0.001),  # 图像透视（+/-分数），范围0-0.001
                'flipud': (1, 0.0, 1.0),  # 图像上下翻转（概率）
                'fliplr': (0, 0.0, 1.0),  # 图像左右翻转（概率）
                'mosaic': (1, 0.0, 1.0),  # 图像马赛克（概率）
                'mixup': (1, 0.0, 1.0)}  # 图像mixup（概率）

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # 仅测试/保存最终epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # 进化索引
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # 保存最佳结果到这里
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # 下载evolve.txt（如果存在）

        for _ in range(300):  # 进化代数
            if Path('evolve.txt').exists():  # 如果evolve.txt存在：选择最佳超参数并突变
                # 选择父代
                parent = 'single'  # 父选择方法：'single'或'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # 考虑的先前结果数量
                x = x[np.argsort(-fitness(x))][:n]  # 前n个突变
                w = fitness(x) - fitness(x).min()  # 权重
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # 随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  # 加权选择
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

                # 突变
                mp, s = 0.8, 0.2  # 突变概率，sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # 增益0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # 突变直到改变（防止重复）
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # 突变

            # 约束到限制
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # 下限
                hyp[k] = min(hyp[k], v[2])  # 上限
                hyp[k] = round(hyp[k], 5)  # 有效数字

            # 训练突变
            results = train(hyp.copy(), opt, device)

            # 写入突变结果
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # 绘制进化
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')