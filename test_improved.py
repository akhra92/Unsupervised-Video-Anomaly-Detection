import pprint
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import datasets
from utils import train_util, log_util, anomaly_util
from config.defaults import _C as config, update_config
from models.improved_astnet import ImprovedASTNet, ImprovedASTNetLarge

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test Improved Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/ped2_wresnet.yaml', type=str)
    parser.add_argument('--model-file', help='model parameters',
                        default='output/final_state_improved.pth', type=str)
    parser.add_argument('--use-multiscale', help='use multi-scale anomaly scoring',
                        default=True, type=bool)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'test_improved')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = [0]

    # Load improved model
    if config.DATASET.DATASET == "ped2":
        model = ImprovedASTNet(config, pretrained=False,
                              use_memory=True,
                              use_temporal_attn=True)
    else:
        model = ImprovedASTNetLarge(config, pretrained=False,
                                   use_memory=True,
                                   use_temporal_attn=True)

    logger.info('Model: {}'.format(model.get_name()))
    model = nn.DataParallel(model, device_ids=gpus).to(device=torch.device('cuda:0'))
    logger.info('Model file: {}'.format(args.model_file))

    # Load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    test_dataset = eval('datasets.get_test_data')(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    mat_loader = datasets.get_label(config)
    mat = mat_loader()

    # Run inference
    if args.use_multiscale:
        logger.info('Using multi-scale anomaly scoring')
        score_list = inference_multiscale(config, test_loader, model)
        assert len(score_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(score_list)} detected videos!'

        # Calculate AUC with multi-scale scoring
        auc, fpr, tpr = calculate_auc_multiscale_custom(config, score_list, mat)
    else:
        logger.info('Using standard PSNR scoring')
        psnr_list = inference(config, test_loader, model)
        assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'

        auc, fpr, tpr = anomaly_util.calculate_auc(config, psnr_list, mat)

    logger.info(f'AUC: {auc * 100:.2f}%')

    # Calculate comprehensive metrics
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df

    all_scores = []
    all_labels = []
    for i in range(len(score_list if args.use_multiscale else psnr_list)):
        if args.use_multiscale:
            scores = score_list[i]
        else:
            scores = anomaly_util.anomaly_score(psnr_list[i],
                                               np.max(psnr_list[i]),
                                               np.min(psnr_list[i]))

        # Normalize
        scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)

        all_scores.extend(scores_norm)
        all_labels.extend(mat[i][fp:])

    import numpy as np
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Comprehensive metrics
    comprehensive = anomaly_util.calculate_comprehensive_metrics(all_labels, all_scores)
    logger.info(f'Average Precision: {comprehensive["AP"] * 100:.2f}%')
    logger.info(f'Equal Error Rate: {comprehensive["EER"] * 100:.2f}%')
    logger.info(f'Best F1 Score: {comprehensive["Best_F1"] * 100:.2f}%')


def inference(config, data_loader, model):
    """Standard inference with PSNR"""
    loss_func_mse = nn.MSELoss(reduction='none')

    model.eval()
    psnr_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print('[{}/{}]'.format(i+1, len(data_loader)))
            psnr_video = []

            video, video_name = train_util.decode_input(input=data, train=False)
            video = [frame.to(device=torch.device('cuda:0')) for frame in video]

            for f in tqdm.tqdm(range(len(video) - fp)):
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]

                # Compute PSNR
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = anomaly_util.psnr_park(mse_imgs)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)

    return psnr_list


def inference_multiscale(config, data_loader, model):
    """Inference with multi-scale anomaly scoring"""
    model.eval()
    scorer = anomaly_util.MultiScaleAnomalyScorer()

    score_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print('[{}/{}]'.format(i+1, len(data_loader)))
            score_video = []

            video, video_name = train_util.decode_input(input=data, train=False)
            video = [frame.to(device=torch.device('cuda:0')) for frame in video]

            for f in tqdm.tqdm(range(len(video) - fp)):
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]

                # Compute multi-scale anomaly score
                anomaly_score = scorer.compute_anomaly_score(output, target.unsqueeze(0))
                score_video.append(anomaly_score)

            score_list.append(score_video)

    return score_list


def calculate_auc_multiscale_custom(config, score_list, mat):
    """Calculate AUC for multi-scale scores"""
    import numpy as np
    from sklearn import metrics

    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df

    scores = np.array([], dtype=np.float)
    labels = np.array([], dtype=np.int)

    for i in range(len(score_list)):
        # Normalize scores for this video
        video_scores = np.array(score_list[i])
        normalized = (video_scores - np.min(video_scores)) / (np.max(video_scores) - np.min(video_scores) + 1e-6)

        scores = np.concatenate((scores, normalized), axis=0)
        labels = np.concatenate((labels, mat[i][fp:]), axis=0)

    assert scores.shape == labels.shape, f'Shape mismatch: scores {scores.shape} vs labels {labels.shape}'

    # Note: For anomaly detection, typically label 1 = anomaly, 0 = normal
    # The original code uses pos_label=0, which might be inverted
    # Check your dataset labeling convention
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr


if __name__ == '__main__':
    main()
