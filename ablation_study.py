"""
Ablation Study Script for Improved ASTNet

This script runs systematic ablation studies to measure the contribution
of each improvement component individually.

Usage:
    python ablation_study.py --cfg config/shanghaitech_wresnet.yaml --ablation all
    python ablation_study.py --cfg config/ped2_wresnet.yaml --ablation temporal_attention
"""

import os
import pprint
import argparse
import json
from datetime import datetime
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn

from config.defaults import _C as config, update_config
from utils import train_util, log_util, loss_util, optimizer_util, anomaly_util
from utils.augmentation_util import VideoAugmentation
from models.improved_astnet import ImprovedASTNet, ImprovedASTNetLarge
import datasets


ABLATION_CONFIGS = {
    'baseline': {
        'name': 'Baseline (Original Model)',
        'use_memory': False,
        'use_temporal_attn': False,
        'use_perceptual': False,
        'use_cbam': False,
        'use_augmentation': False,
        'use_cosine_lr': False,
        'description': 'Original model without any improvements'
    },
    'temporal_attention': {
        'name': 'Baseline + Temporal Attention',
        'use_memory': False,
        'use_temporal_attn': True,
        'use_perceptual': False,
        'use_cbam': False,
        'use_augmentation': False,
        'use_cosine_lr': False,
        'description': 'Add temporal attention module only'
    },
    'memory': {
        'name': 'Baseline + Memory Module',
        'use_memory': True,
        'use_temporal_attn': False,
        'use_perceptual': False,
        'use_cbam': False,
        'use_augmentation': False,
        'use_cosine_lr': False,
        'description': 'Add memory module only'
    },
    'cbam': {
        'name': 'Baseline + CBAM Attention',
        'use_memory': False,
        'use_temporal_attn': False,
        'use_perceptual': False,
        'use_cbam': True,
        'use_augmentation': False,
        'use_cosine_lr': False,
        'description': 'Add CBAM (channel+spatial) attention only'
    },
    'perceptual': {
        'name': 'Baseline + Perceptual Loss',
        'use_memory': False,
        'use_temporal_attn': False,
        'use_perceptual': True,
        'use_cbam': False,
        'use_augmentation': False,
        'use_cosine_lr': False,
        'description': 'Add perceptual loss only'
    },
    'augmentation': {
        'name': 'Baseline + Data Augmentation',
        'use_memory': False,
        'use_temporal_attn': False,
        'use_perceptual': False,
        'use_cbam': False,
        'use_augmentation': True,
        'use_cosine_lr': False,
        'description': 'Add data augmentation only'
    },
    'lr_scheduler': {
        'name': 'Baseline + Cosine LR Scheduler',
        'use_memory': False,
        'use_temporal_attn': False,
        'use_perceptual': False,
        'use_cbam': False,
        'use_augmentation': False,
        'use_cosine_lr': True,
        'description': 'Add cosine annealing LR scheduler only'
    },
    'temporal_memory': {
        'name': 'Temporal Attention + Memory',
        'use_memory': True,
        'use_temporal_attn': True,
        'use_perceptual': False,
        'use_cbam': False,
        'use_augmentation': False,
        'use_cosine_lr': False,
        'description': 'Combined temporal attention and memory'
    },
    'attention_modules': {
        'name': 'All Attention Modules',
        'use_memory': False,
        'use_temporal_attn': True,
        'use_cbam': True,
        'use_perceptual': False,
        'use_augmentation': False,
        'use_cosine_lr': False,
        'description': 'Temporal attention + CBAM'
    },
    'loss_improvements': {
        'name': 'All Loss Improvements',
        'use_memory': False,
        'use_temporal_attn': False,
        'use_perceptual': True,
        'use_cbam': False,
        'use_augmentation': False,
        'use_cosine_lr': False,
        'description': 'Perceptual loss + other loss enhancements'
    },
    'training_improvements': {
        'name': 'All Training Improvements',
        'use_memory': False,
        'use_temporal_attn': False,
        'use_perceptual': False,
        'use_cbam': False,
        'use_augmentation': True,
        'use_cosine_lr': True,
        'description': 'Data augmentation + better LR schedule'
    },
    'full_model': {
        'name': 'Full Model (All Improvements)',
        'use_memory': True,
        'use_temporal_attn': True,
        'use_perceptual': True,
        'use_cbam': True,
        'use_augmentation': True,
        'use_cosine_lr': True,
        'description': 'All improvements combined'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation Study for Improved ASTNet')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)
    parser.add_argument('--ablation', help='ablation configuration name or "all"',
                        default='all', type=str)
    parser.add_argument('--epochs', help='number of training epochs',
                        default=50, type=int)
    parser.add_argument('--output-dir', help='output directory for results',
                        default='ablation_results', type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def run_ablation_experiment(config, ablation_config, output_dir, epochs):
    """
    Run a single ablation experiment with specified configuration

    Returns:
        dict: Results including AUC, training time, etc.
    """
    print(f"\n{'='*80}")
    print(f"Running Ablation: {ablation_config['name']}")
    print(f"Description: {ablation_config['description']}")
    print(f"{'='*80}\n")

    # Create experiment-specific output directory
    exp_name = ablation_config['name'].replace(' ', '_').replace('+', 'plus')
    exp_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    # Setup logger
    logger, _, _ = log_util.create_logger(config, config_file='ablation', phase=exp_name)

    # Build model with ablation configuration
    if config.DATASET.DATASET == "ped2":
        model = ImprovedASTNet(config, pretrained=True,
                              use_memory=ablation_config['use_memory'],
                              use_temporal_attn=ablation_config['use_temporal_attn'])
    else:
        model = ImprovedASTNetLarge(config, pretrained=True,
                                   use_memory=ablation_config['use_memory'],
                                   use_temporal_attn=ablation_config['use_temporal_attn'])

    # Disable CBAM if needed (requires model modification)
    if not ablation_config['use_cbam']:
        # Replace CBAM with identity modules
        replace_cbam_with_identity(model)

    model = nn.DataParallel(model, device_ids=[0]).cuda()

    # Setup loss function
    if ablation_config['use_perceptual']:
        losses = loss_util.ImprovedMultiLossFunction(config=config, use_perceptual=True).cuda()
    else:
        losses = loss_util.MultiLossFunction(config=config).cuda()

    # Setup optimizer
    optimizer = optimizer_util.get_optimizer(config, model)

    # Setup LR scheduler
    if ablation_config['use_cosine_lr']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = optimizer_util.get_scheduler(config, optimizer)

    # Setup data loader
    train_dataset = eval('datasets.get_data')(config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )

    # Training loop (simplified for ablation study)
    logger.info(f'Training for {epochs} epochs...')
    import time
    start_time = time.time()

    for epoch in range(epochs):
        train_epoch(config, train_loader, model, losses, optimizer, epoch, logger,
                   use_augmentation=ablation_config['use_augmentation'])
        scheduler.step()

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.module.state_dict(),
                      os.path.join(exp_output_dir, f'epoch_{epoch+1}.pth'))

    training_time = time.time() - start_time

    # Save final model
    final_model_path = os.path.join(exp_output_dir, 'final_model.pth')
    torch.save(model.module.state_dict(), final_model_path)

    logger.info(f'Training completed in {training_time:.2f} seconds')
    logger.info(f'Model saved to {final_model_path}')

    # Evaluate on test set
    logger.info('Evaluating on test set...')
    test_results = evaluate_model(config, model, logger)

    # Compile results
    results = {
        'ablation_name': ablation_config['name'],
        'description': ablation_config['description'],
        'config': ablation_config,
        'training_time': training_time,
        'epochs': epochs,
        'model_path': final_model_path,
        **test_results
    }

    return results


def train_epoch(config, train_loader, model, loss_functions, optimizer, epoch, logger,
               use_augmentation=False):
    """Train for one epoch"""
    loss_func_mse = nn.MSELoss(reduction='none')
    model.train()

    total_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, target = train_util.decode_input(input=data, train=True)
        output = model(inputs)

        target = target.cuda(non_blocking=True)
        loss = loss_functions(output, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % config.PRINT_FREQ == 0:
            mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
            psnr = anomaly_util.psnr_park(mse_imgs)
            logger.info(f'Epoch [{epoch+1}][{i+1}/{len(train_loader)}] '
                       f'Loss: {loss.item():.4f} PSNR: {psnr:.2f}')

    avg_loss = total_loss / len(train_loader)
    logger.info(f'Epoch [{epoch+1}] Average Loss: {avg_loss:.4f}')


def evaluate_model(config, model, logger):
    """Evaluate model on test set"""
    test_dataset = eval('datasets.get_test_data')(config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    mat_loader = datasets.get_label(config)
    mat = mat_loader()

    # Run inference
    loss_func_mse = nn.MSELoss(reduction='none')
    model.eval()
    psnr_list = []

    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            psnr_video = []
            video, video_name = train_util.decode_input(input=data, train=False)
            video = [frame.cuda() for frame in video]

            for f in range(len(video) - fp):
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]

                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = anomaly_util.psnr_park(mse_imgs)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)

    # Calculate AUC
    auc, fpr, tpr = anomaly_util.calculate_auc(config, psnr_list, mat)

    logger.info(f'Test AUC: {auc * 100:.2f}%')

    return {
        'auc': auc * 100,
        'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
        'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else tpr
    }


def replace_cbam_with_identity(model):
    """Replace CBAM modules with identity mappings for ablation"""
    for name, module in model.named_modules():
        if 'cbam' in name.lower():
            # Replace with identity
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.module if hasattr(model, 'module') else model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, nn.Identity())


def main():
    args = parse_args()
    update_config(config, args)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Determine which ablations to run
    if args.ablation == 'all':
        ablation_names = list(ABLATION_CONFIGS.keys())
    else:
        ablation_names = [args.ablation]
        if args.ablation not in ABLATION_CONFIGS:
            print(f"Error: Unknown ablation '{args.ablation}'")
            print(f"Available ablations: {', '.join(ABLATION_CONFIGS.keys())}")
            return

    print(f"\nRunning {len(ablation_names)} ablation experiments:")
    for name in ablation_names:
        print(f"  - {ABLATION_CONFIGS[name]['name']}")
    print()

    # Run ablation studies
    all_results = []
    for ablation_name in ablation_names:
        ablation_config = ABLATION_CONFIGS[ablation_name]

        try:
            results = run_ablation_experiment(
                config, ablation_config, output_dir, args.epochs
            )
            all_results.append(results)

            # Save individual result
            result_file = os.path.join(output_dir, f'{ablation_name}_results.json')
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n✓ Completed: {ablation_config['name']}")
            print(f"  AUC: {results['auc']:.2f}%")
            print(f"  Training time: {results['training_time']:.2f}s")

        except Exception as e:
            print(f"\n✗ Failed: {ablation_config['name']}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()

    # Save summary results
    summary_file = os.path.join(output_dir, 'ablation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset': config.DATASET.DATASET,
            'epochs': args.epochs,
            'results': all_results
        }, f, indent=2)

    # Print comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Configuration':<40} {'AUC':<10} {'Improvement':<15} {'Time(s)':<10}")
    print("-"*80)

    # Find baseline result
    baseline_auc = None
    for result in all_results:
        if 'Baseline' in result['ablation_name']:
            baseline_auc = result['auc']
            break

    for result in all_results:
        auc = result['auc']
        improvement = f"+{auc - baseline_auc:.2f}%" if baseline_auc else "N/A"
        print(f"{result['ablation_name']:<40} {auc:<10.2f} {improvement:<15} {result['training_time']:<10.2f}")

    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary file: {summary_file}")


if __name__ == '__main__':
    main()
