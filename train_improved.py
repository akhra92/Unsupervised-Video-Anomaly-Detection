import os
import pprint
import argparse
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn

from config.defaults import _C as config, update_config
from utils import train_util, log_util, loss_util, optimizer_util, anomaly_util
from utils.augmentation_util import VideoAugmentation
import models as models
from models.improved_astnet import ImprovedASTNet, ImprovedASTNetLarge
from models.lightweight_astnet import LightweightASTNet
import datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Improved ASTNet Training')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)
    parser.add_argument('--encoder', help='encoder type: wideresnet, efficientnet, mobilenet_large, mobilenet_small',
                        default='wideresnet', type=str)
    parser.add_argument('--use-memory', help='use memory module',
                        default=True, type=bool)
    parser.add_argument('--use-temporal-attn', help='use temporal attention',
                        default=True, type=bool)
    parser.add_argument('--use-perceptual', help='use perceptual loss',
                        default=True, type=bool)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'train_improved')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # Select model based on encoder type
    encoder_type = args.encoder.lower()

    if encoder_type == 'wideresnet':
        # Original WideResNet-based model
        if config.DATASET.DATASET == "ped2":
            model = ImprovedASTNet(config, pretrained=True,
                                  use_memory=args.use_memory,
                                  use_temporal_attn=args.use_temporal_attn)
        else:
            model = ImprovedASTNetLarge(config, pretrained=True,
                                       use_memory=args.use_memory,
                                       use_temporal_attn=args.use_temporal_attn)
    else:
        # Lightweight encoder-based model
        model = LightweightASTNet(config, encoder_type=encoder_type, pretrained=True,
                                 use_memory=args.use_memory,
                                 use_temporal_attn=args.use_temporal_attn)

    logger.info('Model: {}'.format(model.get_name()))
    logger.info(f'Encoder: {encoder_type}')
    logger.info(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    gpus = [0]
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Use improved loss function with perceptual loss
    losses = loss_util.ImprovedMultiLossFunction(config=config,
                                                 use_perceptual=args.use_perceptual).cuda()
    logger.info(f'Using improved loss function (perceptual={args.use_perceptual})')

    optimizer = optimizer_util.get_optimizer(config, model)

    # NEW: Better learning rate scheduler (Cosine Annealing with Warm Restarts)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,      # Restart every 20 epochs
        T_mult=2,    # Double the period after each restart
        eta_min=1e-6
    )
    logger.info('Using CosineAnnealingWarmRestarts scheduler')

    # Load dataset with augmentation
    train_dataset = eval('datasets.get_data')(config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )
    logger.info('Number videos: {}'.format(len(train_dataset)))

    # Initialize augmentation
    augmenter = VideoAugmentation(config, train=True)
    logger.info('Video augmentation enabled')

    last_epoch = config.TRAIN.BEGIN_EPOCH
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        train(config, train_loader, model, losses, optimizer, epoch, logger, augmenter)

        scheduler.step()

        if (epoch + 1) % config.SAVE_CHECKPOINT_FREQ == 0:
            logger.info('=> saving model state epoch_{}.pth to {}\n'.format(epoch+1, final_output_dir))
            torch.save(model.module.state_dict(), os.path.join(final_output_dir,
                                                               'epoch_{}.pth'.format(epoch + 1)))

    final_model_state_file = os.path.join(final_output_dir, 'final_state_improved.pth')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


def train(config, train_loader, model, loss_functions, optimizer, epoch, logger, augmenter=None):
    loss_func_mse = nn.MSELoss(reduction='none')

    model.train()

    for i, data in enumerate(train_loader):
        # Decode input
        inputs, target = train_util.decode_input(input=data, train=True)

        # Apply data augmentation if provided
        # Note: augmentation should be applied before tensor conversion in dataset
        # This is a placeholder - actual augmentation is in the dataset loader

        output = model(inputs)

        # Compute loss
        target = target.cuda(non_blocking=True)
        loss = loss_functions(output, target)

        # Compute PSNR for monitoring
        mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
        psnr = anomaly_util.psnr_park(mse_imgs)

        # Optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        cur_lr = optimizer.param_groups[0]['lr']

        # Log detailed loss components every N iterations
        if (i + 1) % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Lr {lr:.6f}\t' \
                  'Loss {total_loss:.4f}\t' \
                  'PSNR {psnr:.2f}'.format(epoch+1, i+1, len(train_loader),
                                             lr=cur_lr,
                                             total_loss=loss,
                                             psnr=psnr)
            logger.info(msg)


if __name__ == '__main__':
    main()
