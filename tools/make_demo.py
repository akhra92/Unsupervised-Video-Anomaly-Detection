"""
Generate a README demo GIF: test video (left) + live anomaly-score curve (right).

Runs the pretrained ASTNet model on one UCSD Ped2 test clip, computes the
per-frame anomaly score, and renders a synchronized two-panel animation.

Usage:
    python tools/make_demo.py --cfg config/ped2_wresnet.yaml \
        --model-file pretrained/ped2.pth --clip 01 --out assets/ped2_demo.gif
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from config.defaults import _C as config
from datasets.video_data import get_transform
from models.wresnet1024_cattn_tsm import ASTNet as get_net1
from models.wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet as get_net2
from utils import anomaly_util

# Ground-truth abnormal frame ranges (1-indexed) per UCSD Ped2 test clip
GT_RANGES = {
    "01": (61, 180), "02": (95, 180), "03": (1, 146), "04": (31, 180),
    "05": (1, 129), "06": (1, 159), "07": (46, 180), "08": (1, 180),
    "09": (1, 120), "10": (1, 150), "11": (1, 180), "12": (88, 180),
}


def load_clip(frames_dir, transform):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    pil = [Image.open(os.path.join(frames_dir, f)).convert("RGB") for f in files]
    tensors = [transform(p).unsqueeze(0) for p in pil]  # each [1,C,H,W]
    return pil, tensors


@torch.no_grad()
def compute_psnr(model, tensors, fp, device):
    loss_mse = nn.MSELoss(reduction="none")
    tensors = [t.to(device) for t in tensors]
    psnr = []
    for f in range(len(tensors) - fp):
        out = model(tensors[f:f + fp])
        target = tensors[f + fp]
        mse = torch.mean(loss_mse((out + 1) / 2, (target + 1) / 2)).item()
        psnr.append(anomaly_util.psnr_park(mse))
    return np.array(psnr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/ped2_wresnet.yaml")
    ap.add_argument("--model-file", default="pretrained/ped2.pth")
    ap.add_argument("--clip", default="01")
    ap.add_argument("--out", default="assets/ped2_demo.gif")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--step", type=int, default=2,
                    help="render every Nth frame (smaller GIF); scores use all frames")
    ap.add_argument("--relu", action="store_true",
                    help="run with plain ReLU instead of SReLU (needed for the original ped2.pth baseline)")
    args = ap.parse_args()

    if args.relu:
        # The public ped2.pth was trained with ReLU; force ReLU so the weights are valid.
        import models.basic_modules as _bm, models.wider_resnet as _wr
        _bm.SReLU.forward = lambda self, x: F.relu(x)
        _wr.SReLU.forward = lambda self, x: F.relu(x)
        print("Activation override: SReLU -> ReLU (baseline weights)")

    config.defrost()
    config.merge_from_file(args.cfg)
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = get_net1 if config.DATASET.DATASET == "ped2" else get_net2
    model = net(config, pretrained=False).to(device).eval()
    state = torch.load(args.model_file, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    print(f"Loaded {args.model_file}")

    ef, df = config.MODEL.ENCODED_FRAMES, config.MODEL.DECODED_FRAMES
    fp = ef + df

    frames_dir = os.path.join(config.DATASET.ROOT, config.DATASET.DATASET,
                              config.DATASET.TESTSET, args.clip)
    transform = get_transform([config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]])
    pil_frames, tensors = load_clip(frames_dir, transform)
    print(f"Clip {args.clip}: {len(pil_frames)} frames")

    psnr = compute_psnr(model, tensors, fp, device)
    # anomaly score in [0,1]: higher = more anomalous, lightly smoothed
    reg = (psnr - psnr.min()) / (psnr.max() - psnr.min() + 1e-8)
    score = gaussian_filter1d(1.0 - reg, sigma=2.0)

    # GT shading aligned to predicted-frame axis (predictions start at frame index fp)
    gs, ge = GT_RANGES.get(args.clip, (0, 0))
    gt_lo, gt_hi = max(gs - 1 - fp, 0), max(ge - fp, 0)

    x = np.arange(len(score))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    frames_out = []
    for t in range(0, len(score), max(args.step, 1)):
        fig, (axv, axp) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                       gridspec_kw={"width_ratios": [1.15, 1]})
        # left: the predicted/target video frame
        axv.imshow(pil_frames[t + fp])
        axv.set_xticks([]); axv.set_yticks([])
        anomalous = gt_lo <= t < gt_hi
        axv.set_title(f"UCSD Ped2 · Test {args.clip}", fontsize=11)
        if anomalous:
            for s in axv.spines.values():
                s.set_color("red"); s.set_linewidth(3)
            axv.text(0.02, 0.06, "ANOMALY", color="white", fontsize=11, weight="bold",
                     transform=axv.transAxes,
                     bbox=dict(facecolor="red", edgecolor="none", pad=2))

        # right: anomaly score growing up to t
        if gt_hi > gt_lo:
            axp.axvspan(gt_lo, gt_hi, color="red", alpha=0.12, label="ground-truth anomaly")
        axp.plot(x[:t + 1], score[:t + 1], color="#1f77b4", lw=2)
        axp.scatter([t], [score[t]], color="#d62728", zorder=5, s=30)
        axp.set_xlim(0, len(score) - 1)
        axp.set_ylim(-0.02, 1.05)
        axp.set_xlabel("frame"); axp.set_ylabel("anomaly score")
        axp.set_title("Predicted anomaly score", fontsize=11)
        axp.legend(loc="upper left", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames_out.append(buf.copy())
        plt.close(fig)

    imageio.mimsave(args.out, frames_out, fps=args.fps, loop=0)
    mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out}  ({len(frames_out)} frames, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
